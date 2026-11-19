//! How far a video BIOS read piece by piece reaches.
//!
//! A discrete NVIDIA GPU of a laptop with switchable graphics has no option
//! ROM of its own to read: the firmware holds its video BIOS and hands it
//! out through the GPU's ACPI `_ROM` method, 4 KiB at a call. Linux's
//! nouveau reads it there (`nvkm/subdev/bios/shadowacpi.c`), which Stela's
//! Linux profile does not build: it has no ACPI. The boot evaluates `_ROM`
//! with its own interpreter and gives the image to Linux as the function's
//! platform ROM (`pci_dev::rom`), which nouveau reads when its other places
//! hold none (`nvbios_platform`) and checks again in full.
//!
//! `_ROM` does not say how long the image is, and what a firmware answers
//! past its end is its own. So the length is read from the images' own
//! headers, by nouveau's rules (`nvkm/subdev/bios/image.c`, `pcir.c`,
//! `npde.c`): each image begins with a ROM signature and points at a PCI
//! data structure with its length and whether it is the last, which an
//! NVIDIA extension (`NPDE`) after that structure overrides.

/// What `_ROM` gives at a call, at most (ACPI 6.5, B.3.3).
pub const CHUNK: usize = 0x1000;
/// The most that is read. NVIDIA's images with their UEFI drivers are a few
/// hundred KiB.
pub const LIMIT: usize = 1 << 20;

/// What the bytes read so far say of the images' extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Extent {
    /// The images end here: this many bytes hold every image the headers
    /// name, or every one before a header nouveau would refuse.
    Complete(usize),
    /// A header that decides lies past the bytes read: at least this many
    /// bytes are needed.
    Needs(usize),
    /// The bytes begin with no image.
    Invalid(&'static str),
}

/// Offsets in an image's header and its PCI data structure.
const POINTER: usize = 0x18;
const STRUCTURE_LENGTH: usize = 0x0a;
const IMAGE_LENGTH: usize = 0x10;
const IMAGE_TYPE: usize = 0x14;
const INDICATOR: usize = 0x15;
/// The PCI data structure as far as nouveau reads it.
const STRUCTURE: usize = 0x16;
/// The NVIDIA extension: its image length, its last-image flag, and how far
/// nouveau reads it.
const NPDE_IMAGE_LENGTH: usize = 0x08;
const NPDE_INDICATOR: usize = 0x0a;
const NPDE: usize = 0x0b;
const LAST: u8 = 0x80;
/// The image type nouveau takes as the last whatever its flag says.
const TYPE_LAST: u8 = 0x70;

fn word(bytes: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// One image's length and whether another follows, or why its header gives
/// neither.
enum Header {
    Image { length: usize, last: bool },
    Needs(usize),
    Refused(&'static str),
}

fn header(bytes: &[u8], base: usize) -> Header {
    if bytes.len() < base + POINTER + 2 {
        return Header::Needs(base + POINTER + 2);
    }
    if !matches!(word(bytes, base), 0xaa55 | 0xbb77 | 0x4e56) {
        return Header::Refused("no ROM signature");
    }
    let pointer = word(bytes, base + POINTER);
    if pointer == 0 {
        return Header::Refused("no PCI data structure");
    }
    let structure = base + pointer;
    if bytes.len() < structure + STRUCTURE {
        return Header::Needs(structure + STRUCTURE);
    }
    if !matches!(
        &bytes[structure..structure + 4],
        b"PCIR" | b"RGIS" | b"NPDS"
    ) {
        return Header::Refused("no PCI data structure signature");
    }
    let mut length = word(bytes, structure + IMAGE_LENGTH) * 512;
    let mut last = bytes[structure + INDICATOR] & LAST != 0;
    if bytes[structure + IMAGE_TYPE] == TYPE_LAST {
        last = true;
    } else {
        // The extension follows the structure, 16-byte aligned.
        let extension = (structure + word(bytes, structure + STRUCTURE_LENGTH) + 0x0f) & !0x0f;
        if bytes.len() < extension + NPDE {
            return Header::Needs(extension + NPDE);
        }
        if &bytes[extension..extension + 4] == b"NPDE" {
            length = word(bytes, extension + NPDE_IMAGE_LENGTH) * 512;
            last = bytes[extension + NPDE_INDICATOR] & LAST != 0;
        }
    }
    if length == 0 {
        return Header::Refused("an image of no length");
    }
    Header::Image { length, last }
}

/// How far the images in `bytes`, read from the ROM's start, reach.
pub fn extent(bytes: &[u8]) -> Extent {
    let mut base = 0;
    loop {
        match header(bytes, base) {
            Header::Image { length, last } => {
                base += length;
                if last {
                    return Extent::Complete(base);
                }
            }
            Header::Needs(bytes) => return Extent::Needs(bytes),
            // nouveau takes the images before one it refuses.
            Header::Refused(_) if base != 0 => return Extent::Complete(base),
            Header::Refused(why) => return Extent::Invalid(why),
        }
    }
}

/// Why a read of the whole ROM ended without an image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The first bytes hold no image.
    Invalid(&'static str),
    /// A piece the headers call for could not be read; so many bytes were.
    Short { read: usize, needed: usize },
    /// The headers name more than [`LIMIT`] bytes.
    TooLong(usize),
}

/// Reads the whole ROM through `chunk`, which gives the piece at an offset
/// that is a multiple of [`CHUNK`]: [`CHUNK`] bytes, fewer at the ROM's end,
/// or none where it has no more. The bytes of every image the headers name.
pub fn read(
    mut chunk: impl FnMut(usize) -> Option<alloc::vec::Vec<u8>>,
) -> Result<alloc::vec::Vec<u8>, Error> {
    let mut bytes = alloc::vec::Vec::new();
    let mut ended = false;
    loop {
        let needed = match extent(&bytes) {
            Extent::Complete(length) if length <= bytes.len() => {
                bytes.truncate(length);
                return Ok(bytes);
            }
            Extent::Complete(length) | Extent::Needs(length) => length,
            Extent::Invalid(why) => return Err(Error::Invalid(why)),
        };
        if needed > LIMIT {
            return Err(Error::TooLong(needed));
        }
        if ended {
            return Err(Error::Short {
                read: bytes.len(),
                needed,
            });
        }
        while bytes.len() < needed {
            // Whole pieces only: a short one is the ROM's last.
            match chunk(bytes.len()) {
                Some(piece) if !piece.is_empty() && piece.len() <= CHUNK => {
                    ended = piece.len() < CHUNK;
                    bytes.extend_from_slice(&piece);
                    if ended {
                        break;
                    }
                }
                _ => {
                    ended = true;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    /// An image of `blocks` 512-byte blocks whose PCI data structure lies at
    /// `pointer`, with an NVIDIA extension when `npde` gives its blocks and
    /// last flag.
    fn image(
        blocks: u16,
        pointer: u16,
        kind: u8,
        last: bool,
        npde: Option<(u16, bool)>,
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; usize::from(npde.map_or(blocks, |(blocks, _)| blocks)) * 512];
        bytes[..2].copy_from_slice(&0xaa55u16.to_le_bytes());
        bytes[0x18..0x1a].copy_from_slice(&pointer.to_le_bytes());
        let at = usize::from(pointer);
        bytes[at..at + 4].copy_from_slice(b"PCIR");
        bytes[at + 0x0a..at + 0x0c].copy_from_slice(&0x18u16.to_le_bytes());
        bytes[at + 0x10..at + 0x12].copy_from_slice(&blocks.to_le_bytes());
        bytes[at + 0x14] = kind;
        bytes[at + 0x15] = if last { 0x80 } else { 0 };
        if let Some((blocks, last)) = npde {
            let at = (at + 0x18 + 0x0f) & !0x0f;
            bytes[at..at + 4].copy_from_slice(b"NPDE");
            bytes[at + 0x08..at + 0x0a].copy_from_slice(&blocks.to_le_bytes());
            bytes[at + 0x0a] = if last { 0x80 } else { 0 };
        }
        bytes
    }

    /// `rom` as `_ROM` hands it out, zeros past its end as many firmwares do.
    fn pieces(rom: &[u8], padded: bool) -> impl FnMut(usize) -> Option<Vec<u8>> + '_ {
        move |offset| {
            assert_eq!(offset % CHUNK, 0);
            let mut piece: Vec<u8> = rom
                .get(offset..)
                .map(|rest| rest[..rest.len().min(CHUNK)].to_vec())
                .unwrap_or_default();
            if padded {
                piece.resize(CHUNK, 0);
            }
            Some(piece)
        }
    }

    #[test]
    fn one_image_ends_where_its_data_structure_says() {
        let rom = image(126, 0x170, 0, true, None);
        assert_eq!(extent(&rom), Extent::Complete(126 * 512));
        assert_eq!(extent(&rom[..0x10]), Extent::Needs(0x1a));
        assert_eq!(extent(&rom[..0x100]), Extent::Needs(0x170 + 0x16));
        // The first piece decides, before the image is read.
        assert_eq!(extent(&rom[..CHUNK]), Extent::Complete(126 * 512));
    }

    #[test]
    fn the_nvidia_extension_overrides_length_and_last() {
        // The legacy image says it is the last and 100 blocks; the
        // extension says 126 and that another follows.
        let mut rom = image(100, 0x170, 0, true, Some((126, false)));
        let legacy = rom.len();
        assert_eq!(legacy, 126 * 512);
        assert_eq!(extent(&rom), Extent::Needs(legacy + 0x1a));
        // The UEFI image after it, the last by its extension.
        rom.extend(image(200, 0x1c, 3, false, Some((200, true))));
        assert_eq!(extent(&rom), Extent::Complete(legacy + 200 * 512));
        // A type 0x70 image is the last whatever it says.
        let mut rom = image(8, 0x40, 0, false, None);
        rom.extend(image(4, 0x40, 0x70, false, None));
        assert_eq!(extent(&rom), Extent::Complete(12 * 512));
    }

    #[test]
    fn a_refused_header_ends_the_images_before_it_and_none_is_invalid() {
        let mut rom = image(8, 0x40, 0, false, None);
        rom.extend(vec![0xffu8; 0x400]);
        assert_eq!(extent(&rom), Extent::Complete(8 * 512));
        assert_eq!(extent(&[0u8; 0x1000]), Extent::Invalid("no ROM signature"));
        let mut bad = image(8, 0x40, 0, true, None);
        bad[0x40] = b'X';
        assert_eq!(
            extent(&bad),
            Extent::Invalid("no PCI data structure signature")
        );
        let mut none = image(8, 0x40, 0, true, None);
        none[0x18..0x1a].fill(0);
        assert_eq!(extent(&none), Extent::Invalid("no PCI data structure"));
        let mut empty = image(8, 0x40, 0, true, None);
        empty[0x50..0x52].fill(0);
        assert_eq!(extent(&empty), Extent::Invalid("an image of no length"));
    }

    #[test]
    fn the_whole_rom_is_read_in_pieces_and_cut_at_its_end() {
        let mut rom = image(100, 0x170, 0, true, Some((126, false)));
        rom.extend(image(200, 0x1c, 3, false, Some((200, true))));
        for padded in [false, true] {
            let mut calls = 0;
            let mut source = pieces(&rom, padded);
            let read = read(|offset| {
                calls += 1;
                source(offset)
            })
            .unwrap();
            assert_eq!(read, rom);
            // 326 blocks are 41 pieces, the last a partial one.
            assert_eq!(calls, rom.len().div_ceil(CHUNK));
        }
    }

    #[test]
    fn a_rom_that_ends_early_or_names_too_much_is_refused() {
        let rom = image(126, 0x170, 0, true, None);
        assert_eq!(
            read(pieces(&rom[..0x3000], false)),
            Err(Error::Short {
                read: 0x3000,
                needed: 126 * 512
            })
        );
        assert_eq!(
            read(|_| None),
            Err(Error::Short {
                read: 0,
                needed: 0x1a
            })
        );
        assert_eq!(
            read(|_| Some(vec![0u8; CHUNK])),
            Err(Error::Invalid("no ROM signature"))
        );
        // 0xffff blocks are just under 32 MiB.
        let huge = image(16, 0x40, 0, true, Some((0xffff, true)));
        assert_eq!(
            read(pieces(&huge[..CHUNK], true)),
            Err(Error::TooLong(0xffff * 512))
        );
        // A piece longer than `_ROM` may give ends the read.
        assert_eq!(
            read(|_| Some(vec![0u8; CHUNK + 1])),
            Err(Error::Short {
                read: 0,
                needed: 0x1a
            })
        );
    }
}
