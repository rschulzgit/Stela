//! The GPU's ACPI data for GSP-RM.
//!
//! Upstream nouveau gives GSP-RM what the firmware says of a GPU, through
//! Linux's ACPI: the `_DOD` display list and the JT and NVOP capabilities in
//! its system info, and the backlight table of the NBCI or NVHG `_DSM`
//! function 0x14 at display init. Stela's Linux has no ACPI, so this
//! evaluates them on the GPU's ACPI companion with the host's own
//! interpreter before the GPU starts and hands them to
//! `stela_linux::backlight`. The companion is the one Linux's
//! `acpi_pci_find_companion` gives the GPU (probe's `gpu_companions`), as
//! `ACPI_HANDLE` names it to nouveau. Without them GSP-RM knows no internal panel and
//! refuses every brightness control (NV_ERR_NOT_SUPPORTED), and a laptop
//! panel the discrete GPU drives, as on a MUX switched to it, stays dark.
use crate::output;
use alloc::{format, string::String, vec::Vec};
use stela_drivers::{
    bus::pci::Address,
    gpu::{Adapter, Vendor},
    platform::acpi::{Argument, Handle, Session, Value},
};
use stela_linux::backlight::Acpi;

/// `D4A50B75-65C7-46F7-BFB7-41514CEA0244` as `ToUUID` lays it out.
const NBCI: [u8; 16] = [
    0x75, 0x0b, 0xa5, 0xd4, 0xc7, 0x65, 0xf7, 0x46, 0xbf, 0xb7, 0x41, 0x51, 0x4c, 0xea, 0x02, 0x44,
];
/// `9D95A0A0-0060-4D48-B34D-7E5FEA129FD4` as `ToUUID` lays it out.
const NVHG: [u8; 16] = [
    0xa0, 0xa0, 0x95, 0x9d, 0x60, 0x00, 0x48, 0x4d, 0xb3, 0x4d, 0x7e, 0x5f, 0xea, 0x12, 0x9f, 0xd4,
];
/// `A486D8F8-0BDA-471B-A72B-6042A6B5BEE0` as `ToUUID` lays it out.
const NVOP: [u8; 16] = [
    0xf8, 0xd8, 0x86, 0xa4, 0xda, 0x0b, 0x1b, 0x47, 0xa7, 0x2b, 0x60, 0x42, 0xa6, 0xb5, 0xbe, 0xe0,
];
/// `CBECA351-067B-4924-9CBD-B46B00B86F34` as `ToUUID` lays it out.
const JT: [u8; 16] = [
    0x51, 0xa3, 0xec, 0xcb, 0x7b, 0x06, 0x24, 0x49, 0x9c, 0xbd, 0xb4, 0x6b, 0x00, 0xb8, 0x6f, 0x34,
];
/// The backlight table function, which nouveau asks NBCI and NVHG at 0x102.
const BACKLIGHT: u64 = 0x14;
/// `NV2080_CTRL_ACPI_DSM_READ_SIZE`, the most GSP-RM takes.
const LIMIT: usize = 0x1000;
/// `NV0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS`.
const DISPLAYS: usize = 16;

/// Read and keep the ACPI data of each NVIDIA adapter's companion, of the
/// `companions` probe found, logging one line for each.
pub(in crate::engine) fn load(
    session: &'static Session,
    adapters: &[Adapter],
    companions: &[(Address, String)],
) {
    for adapter in adapters.iter().filter(|a| a.vendor == Vendor::Nvidia) {
        let address = adapter.address;
        let found = companions
            .iter()
            .find(|(function, _)| *function == address)
            .and_then(|(_, path)| Some((path, session.handle(path).ok()?)));
        let Some((path, device)) = found else {
            output::arguments(format_args!(
                "GPU {address}: no ACPI companion; GSP-RM gets no ACPI data\n"
            ));
            continue;
        };
        // A GPU before Turing has no GSP-RM to take the rest; nouveau needs
        // its video BIOS, which only the companion's `_ROM` may hold.
        super::acpi_rom::keep(adapter, path, device);
        let (backlight, from) = match table(device) {
            Ok(found) => found,
            Err(why) => (Vec::new(), why),
        };
        let acpi = Acpi {
            backlight,
            dod: dod(device),
            jt: word(device, &JT, 0x103, 1, None),
            nvop: word(device, &NVOP, 0x100, 0x1a, Some(0x1a)),
        };
        output::arguments(format_args!(
            "GPU {address}: ACPI {path}: backlight table {} bytes ({from}), _DOD {:x?}, \
             JT {:x?}, NVOP {:x?}\n",
            acpi.backlight.len(),
            acpi.dod,
            acpi.jt,
            acpi.nvop
        ));
        // Its power, and its port's, as the firmware reports them: Stela
        // runs no ACPI events, so whatever powers the GPU off at run time
        // is the firmware's own doing, and a loss's lines compare with this.
        let port = path
            .rsplit_once('.')
            .map(|(port, _)| port)
            .and_then(|port| Some((port, session.handle(port).ok()?)));
        output::arguments(format_args!(
            "GPU {address}: ACPI power {path}: {}; port {}\n",
            power(device),
            port.map_or_else(
                || String::from("none in the namespace"),
                |(port, handle)| format!("{port}: {}", power(handle))
            )
        ));
        if !stela_linux::backlight::provide(
            address.segment,
            address.bus,
            address.device,
            address.function,
            acpi,
        ) {
            output::arguments(format_args!(
                "GPU {address}: ACPI data not kept: every GPU's place is taken\n"
            ));
        }
    }
}

/// A device's power as Linux's `acpi_bus_get_power_flags` and
/// `acpi_bus_init_power` read it: `_PSC`, each `_PR0` and `_PR3` power
/// resource with its `_STA`, and which of `_PS0`, `_PS3` and `_PRW` it has.
/// Only those and the resources' `_STA` are evaluated.
fn power(device: Handle<'static>) -> String {
    let has = |name: &str| device.child(name).is_ok();
    let mut text = match integer(device, "_PSC") {
        Some(state) => format!("_PSC D{state}"),
        None if has("_PSC") => String::from("_PSC failed"),
        None => String::from("no _PSC"),
    };
    for list in ["_PR0", "_PR3"] {
        if !has(list) {
            continue;
        }
        text.push_str(&format!(", {list} ["));
        match device.evaluate(Some(list), &[]) {
            Ok(Some(Value::Package(resources))) => {
                for (index, resource) in resources.iter().enumerate() {
                    if index != 0 {
                        text.push_str(", ");
                    }
                    let Value::Reference(resource) = resource else {
                        text.push_str("not a reference");
                        continue;
                    };
                    let name = resource.handle.path().unwrap_or_default();
                    match integer(resource.handle, "_STA") {
                        Some(0) => text.push_str(&format!("{name} off")),
                        Some(_) => text.push_str(&format!("{name} on")),
                        None => text.push_str(&format!("{name} _STA failed")),
                    }
                }
            }
            Ok(_) => text.push_str("not a package"),
            Err(error) => text.push_str(&format!("failed: {error:?}")),
        }
        text.push(']');
    }
    for method in ["_PS0", "_PS3", "_PRW"] {
        if has(method) {
            text.push_str(", ");
            text.push_str(method);
        }
    }
    text
}

/// An integer object or method's value, `None` where it is missing or
/// gives something else.
fn integer(handle: Handle<'_>, name: &str) -> Option<u64> {
    handle.child(name).ok()?;
    match handle.evaluate(Some(name), &[]).ok()? {
        Some(Value::Integer(value)) => Some(value),
        _ => None,
    }
}

/// The backlight table and which GUID gave it, or why there is none.
/// nouveau asks only a companion with a `_DSM`.
fn table(device: Handle<'static>) -> Result<(Vec<u8>, String), String> {
    if device.child("_DSM").is_err() {
        return Err(String::from("no _DSM"));
    }
    for (name, guid) in [("NBCI", &NBCI), ("NVHG", &NVHG)] {
        if !offers(device, guid, 0x102, BACKLIGHT) {
            continue;
        }
        let value = dsm(device, guid, 0x102, BACKLIGHT)
            .map_err(|error| format!("{name} _DSM 0x14: {error}"))?;
        return match value.as_ref().and_then(flatten) {
            Some(bytes) if !bytes.is_empty() && bytes.len() <= LIMIT => {
                Ok((bytes, format!("{name} _DSM 0x14")))
            }
            Some(bytes) => Err(format!("{name} _DSM 0x14 gave {} bytes", bytes.len())),
            None => Err(format!("{name} _DSM 0x14 gave no package or buffer")),
        };
    }
    Err(String::from("no NBCI or NVHG _DSM 0x14"))
}

/// `_DOD`'s display IDs, as `r535_gsp_acpi_dod` reads them.
fn dod(device: Handle<'static>) -> Option<Vec<u32>> {
    let Ok(Some(Value::Package(elements))) = device.evaluate(Some("_DOD"), &[]) else {
        return None;
    };
    if elements.len() > DISPLAYS {
        return None;
    }
    elements
        .iter()
        .map(|element| match element {
            Value::Integer(id) => u32::try_from(*id).ok(),
            _ => None,
        })
        .collect()
}

/// A `_DSM` function's 4-byte buffer as a word, as `r535_gsp_acpi_jt` and
/// `r535_gsp_acpi_caps` read them; `check` first asks function 0 for that
/// function where nouveau does.
fn word(
    device: Handle<'static>,
    guid: &[u8; 16],
    revision: u64,
    function: u64,
    check: Option<u64>,
) -> Option<u32> {
    if let Some(check) = check
        && !offers(device, guid, revision, check)
    {
        return None;
    }
    match dsm(device, guid, revision, function).ok()? {
        Some(Value::Buffer(bytes)) => Some(u32::from_le_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

/// `_DSM` with a zeroed 4-byte buffer as Arg3, where nouveau passes a
/// buffer of its own.
fn dsm(
    device: Handle<'static>,
    guid: &[u8; 16],
    revision: u64,
    function: u64,
) -> Result<Option<Value<'static>>, String> {
    device
        .evaluate(
            Some("_DSM"),
            &[
                Argument::Buffer(guid),
                Argument::Integer(revision),
                Argument::Integer(function),
                Argument::Buffer(&[0; 4]),
            ],
        )
        .map_err(|error| format!("{error:?}"))
}

/// Linux's `acpi_check_dsm` for one function: function 0's answer, a
/// buffer or integer bit mask, has its bit set.
fn offers(device: Handle<'static>, guid: &[u8; 16], revision: u64, function: u64) -> bool {
    match device.dsm(guid, revision, 0) {
        Ok(Some(Value::Buffer(mask))) => mask
            .get((function / 8) as usize)
            .is_some_and(|byte| byte & (1 << (function % 8)) != 0),
        Ok(Some(Value::Integer(mask))) => function < 64 && mask & (1 << function) != 0,
        _ => false,
    }
}

/// The table as nouveau copies it: a package's integers, each 4 bytes, or 8
/// when past 32 bits, little-endian; a buffer as it is.
fn flatten(value: &Value<'_>) -> Option<Vec<u8>> {
    match value {
        Value::Buffer(bytes) => Some(bytes.clone()),
        Value::Package(elements) => {
            let mut bytes = Vec::new();
            for element in elements {
                let Value::Integer(word) = element else {
                    return None;
                };
                if word >> 32 != 0 {
                    bytes.extend_from_slice(&word.to_le_bytes());
                } else {
                    bytes.extend_from_slice(&(*word as u32).to_le_bytes());
                }
            }
            Some(bytes)
        }
        _ => None,
    }
}
