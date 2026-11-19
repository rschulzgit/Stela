//! Opens the Metal service for one GPU through its [`GpuStack`]: admits its
//! Vulkan device, derives the platform facts Vulkan does not report, and
//! serves a [`Device`] over the backend and compiler. [`open`] composes the
//! native stack: the stack's Vulkan driver through [`NativeBackend`], which
//! builds its pipelines from the AIR compiler service's internal kernels, and
//! an [`AirCompiler`] for the device the driver reports.
#[cfg(feature = "metal-vulkan")]
use crate::metal::{
    compiler::{self, AirCompiler},
    native::{InternalKernels, Loader, MemoryExporter, NativeBackend, vk},
};
use crate::{
    metal::{
        capabilities::Platform,
        command_buffer::Segment,
        device::Device,
        error::Error,
        nvidia,
        service::{Service, identity::DeviceEpoch},
        shader::Compiler,
        vulkan::{
            MemoryHeapFlags, MemoryProperties,
            backend::{Backend, Object},
            physical::{Identity, PciBusInfo, PhysicalDevice},
        },
    },
    stack::GpuStack,
};
#[cfg(feature = "metal-vulkan")]
use alloc::boxed::Box;
use alloc::{format, string::String};

/// The negotiated PCI Express link, from the Link Status register
/// (`PCI_EXP_LNKSTA`) of the GPU's PCIe capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcieLink {
    /// Current Link Speed: 1 for 2.5 GT/s through 6 for 64 GT/s.
    pub speed: u8,
    /// Negotiated Link Width in lanes.
    pub width: u8,
}

impl PcieLink {
    /// No link: a function on the processor's own fabric, such as an
    /// integrated GPU, which Metal reports with no transfer rate.
    pub const NONE: Self = Self { speed: 0, width: 0 };

    /// The link a GPU is admitted with: the one its PCIe capability reports,
    /// or [`Self::NONE`] for a GPU its stack reports as integrated, whose
    /// function has no such capability. A discrete GPU without one is
    /// refused with the reason.
    pub fn admitted(link: Option<Self>, integrated: bool) -> Result<Self, &'static str> {
        match (link, integrated) {
            (Some(link), _) => Ok(link),
            (None, true) => Ok(Self::NONE),
            (None, false) => Err("no PCIe link"),
        }
    }

    pub const fn from_link_status(status: u16) -> Self {
        Self {
            speed: (status & 0xf) as u8,
            width: ((status >> 4) & 0x3f) as u8,
        }
    }

    /// Data bandwidth in bytes per second after line encoding, as Linux's
    /// `PCIE_SPEED2MBS_ENC`: 8b/10b through 5 GT/s, 128b/130b from 8 GT/s,
    /// and FLIT mode at 64 GT/s. Unknown speeds report zero.
    pub const fn bytes_per_second(self) -> u64 {
        let megabits_per_lane: u64 = match self.speed {
            1 => 2500 * 8 / 10,
            2 => 5000 * 8 / 10,
            3 => 8000 * 128 / 130,
            4 => 16000 * 128 / 130,
            5 => 32000 * 128 / 130,
            6 => 64000,
            _ => 0,
        };
        megabits_per_lane * self.width as u64 * 1_000_000 / 8
    }
}

/// Where the GPU is attached, as the platform's PCI and ACPI owners report it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Attachment {
    pub link: PcieLink,
    /// Behind a hot-pluggable Thunderbolt or USB4 tunnel.
    pub external: bool,
    /// The slot or enclosure index Metal reports as `locationNumber`.
    pub location_number: u64,
}

/// A stack's admission of its Vulkan device as a Metal device.
pub struct Admission {
    /// Admits the Vulkan device `identity` whose kernel driver reports the
    /// chip ([`crate::stack::Kernel::chip`]), and answers the facts Vulkan
    /// does not report for it.
    pub admit: fn(&Identity, u32, Attachment) -> Result<Platform, Refusal>,
}

/// Why a stack did not admit a Vulkan device, in its vendor's terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refusal {
    Nvidia(nvidia::Refusal),
    #[cfg(feature = "amdgpu-port")]
    Amd(crate::metal::amd::Refusal),
    #[cfg(feature = "xe-port")]
    Intel(crate::metal::intel::Refusal),
}

/// The GPU the platform hands the Metal device.
#[derive(Clone, Copy)]
pub struct Gpu {
    pub stack: &'static GpuStack,
    pub pci: PciBusInfo,
    /// The kernel driver's identifier of the chip.
    pub chip: u32,
    pub attachment: Attachment,
}

/// Why the Metal service did not open.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenError {
    /// The device is not one its stack drives.
    Refused(Refusal),
    /// The device lacks a requirement or could not be created.
    Device(Error),
}

impl From<Error> for OpenError {
    fn from(error: Error) -> Self {
        Self::Device(error)
    }
}

/// Serves the Metal device over `backend`, which drives `gpu`, and
/// `compiler`, in the gate's `epoch` for it.
pub fn serve<B: Backend, C: Compiler>(
    backend: B,
    compiler: C,
    gpu: &Gpu,
    epoch: DeviceEpoch,
) -> Result<Service<B, C>, OpenError> {
    Ok(Service::new(device(backend, compiler, gpu)?, epoch))
}

/// The Metal device over `backend`, once it is admitted as `gpu`.
fn device<B: Backend, C: Compiler>(
    backend: B,
    compiler: C,
    gpu: &Gpu,
) -> Result<Device<B, C>, OpenError> {
    let identity = &backend.physical_device().identity;
    if identity.pci_bus != Some(gpu.pci) {
        return Err(Error::Invalid("the backend drives another GPU").into());
    }
    let admission = gpu.stack.metal.as_ref().ok_or(Error::Invalid(
        "the GPU's stack drives displays only: it has no Metal admission",
    ))?;
    let platform =
        (admission.admit)(identity, gpu.chip, gpu.attachment).map_err(OpenError::Refused)?;
    Ok(Device::new(backend, compiler, &platform)?)
}

/// What the Vulkan driver reports of `physical`, for the boot log:
/// `VULKAN DEVICE …`, which is all a boot shows of a GPU without display.
pub fn describe(physical: &PhysicalDevice) -> String {
    let identity = &physical.identity;
    let api = identity.api_version;
    let mut line = format!(
        "VULKAN DEVICE \"{}\" {:04x}:{:04x} {:?} {:?} vulkan={}.{}.{} subgroup={}..{} heaps:",
        identity.device_name,
        identity.vendor_id,
        identity.device_id,
        identity.driver_id,
        identity.device_type,
        api >> 22,
        (api >> 12) & 0x3ff,
        api & 0xfff,
        physical.limits.min_subgroup_size,
        physical.limits.max_subgroup_size,
    );
    for heap in &physical.memory.heaps {
        let local = if heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL) {
            "local"
        } else {
            "host"
        };
        line.push_str(&format!(" {}M/{local}", heap.size >> 20));
    }
    let visible = physical.memory.types.iter().any(|memory| {
        memory
            .properties
            .contains(MemoryProperties::DEVICE_LOCAL | MemoryProperties::HOST_VISIBLE)
    });
    line.push_str(if visible {
        " host-visible-vram=yes"
    } else {
        " host-visible-vram=no"
    });
    // Every requirement of the Metal device the Vulkan device lacks, where
    // the device's own refusal names only the first.
    let missing: alloc::vec::Vec<&str> = crate::metal::vulkan::backend::requirements(physical)
        .iter()
        .filter(|requirement| !requirement.present)
        .map(|requirement| requirement.name)
        .collect();
    if !missing.is_empty() {
        line.push_str(" missing-requirements=");
        line.push_str(&missing.join(","));
    }
    line
}

/// How long a first submission may take to signal: a GPU whose rings do
/// not run never does, and the boot goes on without its Metal device.
const FIRST_SUBMISSION_NS: u64 = 2_000_000_000;

/// Submits one empty command buffer that signals a timeline semaphore and
/// waits for it: the first time the kernel driver's command submission, its
/// fence and its interrupt run for the device. Whether it signaled in time.
pub fn first_submission<B: Backend>(backend: &mut B) -> Result<bool, Error> {
    let semaphore = backend.create_semaphore(0)?;
    let segment = Segment {
        waits: alloc::vec::Vec::new(),
        binds: alloc::vec::Vec::new(),
        commands: alloc::vec::Vec::new(),
        signals: alloc::vec![(semaphore, 1)],
    };
    let signaled = backend
        .submit(core::slice::from_ref(&segment))
        .and_then(|()| backend.wait_semaphore(semaphore, 1, FIRST_SUBMISSION_NS))
        // A submission the kernel driver refused still signals, on the
        // driver's submit thread after the call returned: its failure is
        // taken by the signal.
        .and_then(|signaled| match signaled {
            true => backend.submit_failure(semaphore, 1).map_or(Ok(true), Err),
            false => Ok(false),
        });
    // A semaphore whose signal never came stays, as the GPU may yet write it.
    if matches!(signaled, Ok(true)) {
        backend.destroy(Object::Semaphore(semaphore));
    }
    signaled
}

/// The Metal service over a linked Vulkan driver and the AIR compiler.
#[cfg(feature = "metal-vulkan")]
pub type NativeService = Service<NativeBackend, AirCompiler>;

/// A Vulkan driver's backend for `target`, with the AIR compiler over the
/// service `compiler` answers for its device.
///
/// # Safety
///
/// `loader` is the driver's entry point and `compiler` answers the AIR compiler
/// service with the driver's NIR options for the device it is given, both
/// outliving what this returns.
#[cfg(feature = "metal-vulkan")]
pub unsafe fn stack(
    loader: Loader,
    pci: PciBusInfo,
    exporter: Box<dyn MemoryExporter>,
    compiler: impl FnOnce(vk::PhysicalDevice) -> compiler::Service,
) -> Result<(NativeBackend, AirCompiler), Error> {
    let mut built = None;
    let kernels = |physical| {
        let service = compiler(physical);
        // SAFETY: the caller guarantees the service and its options.
        let kernels = unsafe { service.internal_kernels() }?;
        let [
            accumulate,
            select,
            resolve_draws,
            resolve_dispatches,
            store_texels,
            load_texels,
            average_samples,
            decode_etc,
            decode_pvrtc,
            reorder_blocks,
            decode_astc,
        ] = kernels.code();
        // The kernels last until the pipelines hold their own copies.
        built = Some((service, kernels));
        Ok(InternalKernels {
            accumulate,
            select,
            resolve_draws,
            resolve_dispatches,
            store_texels,
            load_texels,
            average_samples,
            decode_etc,
            decode_pvrtc,
            reorder_blocks,
            decode_astc,
        })
    };
    // SAFETY: the caller guarantees the entry point, and the kernels are the
    // service's NIR for the driver.
    let backend = unsafe { NativeBackend::new(loader, pci, exporter, kernels) }?;
    let Some((service, kernels)) = built else {
        return Err(Error::Invalid("the backend asked for no kernels"));
    };
    drop(kernels);
    // SAFETY: the caller guarantees the service and its options outlive it.
    let compiler = unsafe { AirCompiler::new(service, backend.physical_device()) };
    Ok((backend, compiler))
}

/// Opens the Vulkan driver's device for `gpu` and serves it with the AIR
/// compiler service `compiler` answers for that device.
///
/// # Safety
///
/// `loader` is the entry point of `gpu`'s stack's driver and `compiler`
/// answers the AIR compiler service with that driver's NIR options for the
/// device it is given, both outliving the service.
#[cfg(feature = "metal-vulkan")]
pub unsafe fn open(
    loader: Loader,
    exporter: Box<dyn MemoryExporter>,
    compiler: impl FnOnce(vk::PhysicalDevice) -> compiler::Service,
    gpu: &Gpu,
    epoch: DeviceEpoch,
) -> Result<NativeService, OpenError> {
    // SAFETY: the caller's guarantees.
    let (mut backend, compiler) = unsafe { stack(loader, gpu.pci, exporter, compiler) }?;
    (loader.log)(&describe(backend.physical_device()));
    if loader.keep_lost {
        let failed = match first_submission(&mut backend) {
            Ok(true) => {
                (loader.log)("first submission signaled");
                None
            }
            Ok(false) => {
                (loader.log)("first submission did not signal within 2 s");
                Some(Error::Unsupported(
                    "a GPU whose first submission never signaled",
                ))
            }
            Err(error) => {
                (loader.log)(&format!("first submission failed: {error:?}"));
                Some(error)
            }
        };
        if let Some(error) = failed {
            // A GPU whose first work hung is one its kernel driver resets,
            // and releasing the device would wait for that reset's end.
            core::mem::forget(backend);
            return Err(error.into());
        }
    }
    serve(backend, compiler, gpu, epoch)
}

/// Opens the driver's device for `gpu` as [`open`] does, for a service whose device
/// was lost ([`Service::with_reopen`]).
///
/// # Safety
///
/// As for [`open`].
#[cfg(feature = "metal-vulkan")]
pub unsafe fn reopen(
    loader: Loader,
    exporter: Box<dyn MemoryExporter>,
    compiler: impl FnOnce(vk::PhysicalDevice) -> compiler::Service,
    gpu: &Gpu,
) -> Result<Device<NativeBackend, AirCompiler>, Error> {
    // SAFETY: the caller's guarantees.
    let (backend, compiler) = unsafe { stack(loader, gpu.pci, exporter, compiler) }?;
    device(backend, compiler, gpu).map_err(|error| match error {
        OpenError::Device(error) => error,
        OpenError::Refused(_) => Error::Invalid("the GPU was refused as it reopened"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{
        device::fixture::{MockBackend, MockCompiler},
        nvidia::CHIPS,
        vulkan::DriverId,
    };

    #[test]
    fn a_gpu_without_a_link_is_admitted_only_when_integrated() {
        let gen4_x16 = PcieLink::from_link_status(4 | 16 << 4);
        assert_eq!(PcieLink::admitted(Some(gen4_x16), false), Ok(gen4_x16));
        // An integrated GPU behind a link keeps what the link reports.
        assert_eq!(PcieLink::admitted(Some(gen4_x16), true), Ok(gen4_x16));
        assert_eq!(PcieLink::admitted(None, true), Ok(PcieLink::NONE));
        assert_eq!(PcieLink::NONE.bytes_per_second(), 0);
        assert_eq!(PcieLink::admitted(None, false), Err("no PCIe link"));
    }
    use crate::stack::NVIDIA;
    use core::num::NonZeroU64;

    fn gpu(chipset: u16) -> Gpu {
        Gpu {
            stack: &NVIDIA,
            pci: PciBusInfo {
                domain: 0,
                bus: 1,
                device: 0,
                function: 0,
            },
            chip: chipset.into(),
            attachment: Attachment {
                link: PcieLink {
                    speed: 4,
                    width: 16,
                },
                external: true,
                location_number: 2,
            },
        }
    }

    fn serve_on(
        backend: MockBackend,
        gpu: &Gpu,
    ) -> Result<Service<MockBackend, MockCompiler>, OpenError> {
        let epoch = DeviceEpoch::from_registered(NonZeroU64::MIN);
        serve(backend, MockCompiler::default(), gpu, epoch)
    }

    #[test]
    fn an_admitted_gpu_is_served_with_its_platform() {
        let mut service = serve_on(MockBackend::new(), &gpu(CHIPS[0].chipset)).unwrap();
        let capabilities = service.device().capabilities();
        assert_eq!(capabilities.architecture, CHIPS[0].name);
        assert!(capabilities.removable);
        assert_eq!(capabilities.location_number, 2);
    }

    /// A stack's admission is what serves its GPU: RADV's integrated
    /// device through the AMD admission, with amdgpu's family name and a
    /// built-in location, and refused by the NVIDIA one.
    #[cfg(feature = "amdgpu-port")]
    #[test]
    fn an_amd_gpu_is_served_through_its_stacks_admission() {
        use crate::metal::{abi::DeviceLocation, amd, vulkan::PhysicalDeviceType};
        use crate::stack::AMD;
        let radv = || {
            let mut backend = MockBackend::new();
            let identity = &mut backend.physical.as_mut().unwrap().identity;
            identity.vendor_id = amd::VENDOR_ID;
            identity.driver_id = DriverId::MesaRadv;
            identity.device_id = 0x13c0;
            identity.device_type = PhysicalDeviceType::IntegratedGpu;
            backend
        };
        let mut raphael = gpu(0);
        raphael.stack = &AMD;
        raphael.chip = 149;
        let mut service = serve_on(radv(), &raphael).unwrap();
        let capabilities = service.device().capabilities();
        assert_eq!(capabilities.architecture, "GC_10_3_6");
        assert_eq!(capabilities.location, DeviceLocation::BuiltIn);
        assert_eq!(capabilities.max_transfer_rate, 0);
        let mut on_nvidia = raphael;
        on_nvidia.stack = &NVIDIA;
        assert_eq!(
            serve_on(radv(), &on_nvidia).err(),
            Some(OpenError::Refused(Refusal::Nvidia(
                nvidia::Refusal::NotNvidia {
                    vendor_id: amd::VENDOR_ID
                }
            )))
        );
        raphael.chip = 0;
        assert_eq!(
            serve_on(radv(), &raphael).err(),
            Some(OpenError::Refused(Refusal::Amd(
                amd::Refusal::UnknownFamily { family: 0 }
            )))
        );
    }

    /// A stack's admission is what serves its GPU: ANV's Tiger Lake device
    /// through the Intel admission, with xe's platform name and a built-in
    /// location, and refused by the NVIDIA one.
    #[cfg(feature = "xe-port")]
    #[test]
    fn an_intel_gpu_is_served_through_its_stacks_admission() {
        use crate::metal::{abi::DeviceLocation, intel, vulkan::PhysicalDeviceType};
        static DRAWING_INTEL: GpuStack = GpuStack {
            vendor: crate::Vendor::Intel,
            kernel: crate::stack::Kernel {
                driver: "xe",
                admits: crate::xe::admits,
                registers: 0,
                dma_bits: crate::xe::dma_bits,
                integrated: crate::xe::integrated,
                probed_integrated: None,
                chip: crate::xe::query::chip,
                memory_bytes: crate::xe::query::memory_bytes,
                reset: None,
            },
            metal: Some(intel::ADMISSION),
        };
        let anv = || {
            let mut backend = MockBackend::new();
            let identity = &mut backend.physical.as_mut().unwrap().identity;
            identity.vendor_id = intel::VENDOR_ID;
            identity.driver_id = DriverId::IntelOpenSourceMesa;
            identity.device_id = 0x9a49;
            identity.device_type = PhysicalDeviceType::IntegratedGpu;
            backend
        };
        let mut tiger_lake = gpu(0);
        tiger_lake.stack = &DRAWING_INTEL;
        tiger_lake.chip = 0x9a49;
        let mut service = serve_on(anv(), &tiger_lake).unwrap();
        let capabilities = service.device().capabilities();
        assert_eq!(capabilities.architecture, "TIGERLAKE");
        assert_eq!(capabilities.location, DeviceLocation::BuiltIn);
        assert_eq!(capabilities.max_transfer_rate, 0);
        let mut on_nvidia = tiger_lake;
        on_nvidia.stack = &NVIDIA;
        assert_eq!(
            serve_on(anv(), &on_nvidia).err(),
            Some(OpenError::Refused(Refusal::Nvidia(
                nvidia::Refusal::NotNvidia {
                    vendor_id: intel::VENDOR_ID
                }
            )))
        );
    }

    #[test]
    fn other_gpus_and_drivers_are_refused() {
        let mut other = gpu(CHIPS[0].chipset);
        other.pci.bus = 2;
        assert_eq!(
            serve_on(MockBackend::new(), &other).err(),
            Some(OpenError::Device(Error::Invalid(
                "the backend drives another GPU"
            )))
        );
        assert_eq!(
            serve_on(MockBackend::new(), &gpu(0x140)).err(),
            Some(OpenError::Refused(Refusal::Nvidia(
                nvidia::Refusal::PreTuring { chipset: 0x140 }
            )))
        );
        let mut backend = MockBackend::new();
        let physical = backend.physical.as_mut().unwrap();
        physical.identity.driver_id = DriverId::MesaRadv;
        assert_eq!(
            serve_on(backend, &gpu(CHIPS[0].chipset)).err(),
            Some(OpenError::Refused(Refusal::Nvidia(
                nvidia::Refusal::NotNvk(DriverId::MesaRadv)
            )))
        );
    }

    #[test]
    fn the_log_names_the_device_its_driver_and_heaps() {
        let mut backend = MockBackend::new();
        let identity = &mut backend.physical.as_mut().unwrap().identity;
        identity.vendor_id = 0x1002;
        identity.device_id = 0x73bf;
        identity.driver_id = DriverId::MesaRadv;
        identity.device_type = crate::metal::vulkan::PhysicalDeviceType::DiscreteGpu;
        identity.device_name = "AMD Radeon RX 6800 XT (RADV NAVI21)".into();
        let line = describe(backend.physical.as_ref().unwrap());
        assert!(line.starts_with(
            "VULKAN DEVICE \"AMD Radeon RX 6800 XT (RADV NAVI21)\" 1002:73bf MesaRadv DiscreteGpu"
        ));
        assert!(line.contains(" heaps: "));
        assert!(!line.contains("missing-requirements"));
        // What an unpatched NVK lacks on Kepler of the requirements, on a
        // driver whose compiler is not known to make up for any of it.
        let features = &mut backend.physical.as_mut().unwrap().features;
        features.shader_float16 = false;
        features.vulkan_memory_model = false;
        features.shader_storage_image_read_without_format = false;
        let line = describe(backend.physical.as_ref().unwrap());
        assert!(line.ends_with(
            " missing-requirements=shaderFloat16,vulkanMemoryModel,storageImageWithoutFormat"
        ));
        // NAK compiles the first two for every shader model.
        backend.physical.as_mut().unwrap().identity.driver_id = DriverId::MesaNvk;
        let line = describe(backend.physical.as_ref().unwrap());
        assert!(line.ends_with(" missing-requirements=storageImageWithoutFormat"));
    }

    #[test]
    fn a_first_submission_that_never_signals_keeps_its_semaphore() {
        // The mock completes nothing on its own, as a hung GPU.
        let mut backend = MockBackend::new();
        assert_eq!(first_submission(&mut backend), Ok(false));
        assert!(backend.freed_semaphores.is_empty());
        assert_eq!(backend.semaphores.len(), 1);
    }
}
