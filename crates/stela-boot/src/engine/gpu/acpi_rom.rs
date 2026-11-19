//! The video BIOS a GPU's ACPI companion hands out through `_ROM`.
//!
//! A laptop's discrete NVIDIA GPU beside an integrated one has no option
//! ROM of its own: the firmware holds its video BIOS, and nouveau reads it
//! through the GPU's ACPI `_ROM` method (`nvkm/subdev/bios/shadowacpi.c`).
//! Stela's Linux has no ACPI, so the boot reads it here with the host's own
//! interpreter, before the GPU starts, and gives it to Linux as the
//! function's platform ROM (`pci_dev::rom`), where nouveau looks when its
//! other places hold none (`nvbios_platform`).
//!
//! A GPU that runs GSP-RM never needs it: its own firmware runs devinit
//! after every reset and leaves the VBIOS where nouveau's first place reads
//! it. A Kepler GPU does. The claim resets the function, after which the
//! image firmware posted it with is gone from the GPU, and nouveau must
//! post it again from the VBIOS, as it does on Linux when such a GPU comes
//! back from D3cold. So only the functions before Turing are read
//! (`nouveau_chip::before_turing`).
use crate::output;
use alloc::{boxed::Box, vec::Vec};
use core::{
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};
use stela_boot_core::option_rom::{self, CHUNK};
use stela_drivers::{
    bus::pci::Address,
    gpu::Adapter,
    platform::acpi::{Argument, Handle, Value},
};

/// Each function's image, kept for the boot. Written before any claim, by
/// the one firmware call that reads the GPUs' ACPI data.
static IMAGES: AtomicPtr<Vec<(Address, Vec<u8>)>> = AtomicPtr::new(ptr::null_mut());

/// Reads the video BIOS of `adapter` from `_ROM` of its ACPI companion
/// `device` at `path` and keeps it, with one line saying what was found.
///
/// Called where the GPUs' other ACPI data is read (`acpi_backlight`), which
/// only an image with a Vulkan driver does.
#[cfg_attr(not(feature = "nvk"), allow(dead_code))]
pub(super) fn keep(adapter: &Adapter, path: &str, device: Handle<'static>) {
    let address = adapter.address;
    if !stela_drivers::gpu::nouveau_chip::before_turing(adapter.device) {
        return;
    }
    if device.child("_ROM").is_err() {
        output::arguments(format_args!(
            "GPU {address}: ACPI {path} has no _ROM: nouveau finds the video BIOS on the GPU \
             itself or not at all\n"
        ));
        return;
    }
    let mut calls = 0u32;
    let mut failure = None;
    let read = option_rom::read(|offset| {
        calls += 1;
        let answer = device.evaluate(
            Some("_ROM"),
            &[
                Argument::Integer(offset as u64),
                Argument::Integer(CHUNK as u64),
            ],
        );
        match answer {
            Ok(Some(Value::Buffer(bytes))) => Some(bytes),
            Ok(_) => {
                failure = Some(alloc::format!("_ROM({offset:#x}) gave no buffer"));
                None
            }
            Err(error) => {
                failure = Some(alloc::format!("_ROM({offset:#x}) failed: {error:?}"));
                None
            }
        }
    });
    match read {
        Ok(image) => {
            let head = &image[..image.len().min(4)];
            output::arguments(format_args!(
                "GPU {address}: video BIOS of {} bytes from ACPI {path}._ROM in {calls} calls \
                 (begins {head:02x?}), given to Linux as its platform ROM\n",
                image.len()
            ));
            // SAFETY: the pointer is this function's own, stored below by an
            // earlier call of the same single firmware call; nothing reads
            // it before the claims, which come after that call.
            let mut images =
                match unsafe { IMAGES.swap(ptr::null_mut(), Ordering::AcqRel).as_mut() } {
                    // SAFETY: made by `Box::into_raw` below and taken out above.
                    Some(images) => unsafe { Box::from_raw(images) },
                    None => Box::default(),
                };
            images.retain(|(kept, _)| *kept != address);
            images.push((address, image));
            IMAGES.store(Box::into_raw(images), Ordering::Release);
        }
        Err(error) => output::arguments(format_args!(
            "GPU {address}: no video BIOS from ACPI {path}._ROM after {calls} calls ({error:?}{}{}): \
             nouveau finds it on the GPU itself or not at all\n",
            if failure.is_some() { "; " } else { "" },
            failure.as_deref().unwrap_or("")
        )),
    }
}

/// The image kept for the function at `address`, or nothing.
pub(super) fn image(address: Address) -> &'static [u8] {
    // SAFETY: stored before any claim and never freed after it.
    let Some(images) = (unsafe { IMAGES.load(Ordering::Acquire).as_ref() }) else {
        return &[];
    };
    images
        .iter()
        .find(|(kept, _)| *kept == address)
        .map_or(&[], |(_, image)| image.as_slice())
}
