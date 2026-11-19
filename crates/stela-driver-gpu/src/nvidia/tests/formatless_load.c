/* SPDX-License-Identifier: GPL-2.0-or-later */
/* A Kepler storage image read without a format unpacks every texel as the
 * read with its format does.
 *
 * NVK lowers Kepler's image loads to raw texel loads that the shader
 * unpacks (nak_nir_lower_image_addrs), by the format the load or its
 * variable names. Stela's patch reads the format from the image's
 * descriptor where neither names one. No device is opened here: for each
 * format a Kepler storage descriptor can hold, and a few texels, the load is
 * lowered once with its format and once without, the descriptor's words and
 * the raw texel loads are replaced by the constants a device would give,
 * and both shaders must fold to the same four words. Two formats are
 * checked against values written out here as well, so the reference itself
 * is not all the test rests on.
 */
#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "nil.h"
#include "nouveau/compiler/nak.h"
#include "nouveau/headers/nv_device_info.h"
#include "nouveau/headers/nvidia/classes/cla097.h"
#include "util/format/u_format.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

/* nil's su_info.rs `format_info`: the hardware's format number and its
 * component description, bits 0-1 the component size (0 is 32 bits, 1 is 16,
 * 2 is 8) and bits 2-3 the component count (0 is one, 1 is two, 2 is four).
 */
static const struct format {
   enum pipe_format format;
   uint8_t hw, description;
} formats[] = {
   { PIPE_FORMAT_R32G32B32A32_FLOAT, 0x02, 0x8 },
   { PIPE_FORMAT_R32G32B32A32_SINT, 0x03, 0x8 },
   { PIPE_FORMAT_R32G32B32A32_UINT, 0x04, 0x8 },
   { PIPE_FORMAT_R16G16B16A16_FLOAT, 0x0c, 0x9 },
   { PIPE_FORMAT_R16G16B16A16_UNORM, 0x08, 0x9 },
   { PIPE_FORMAT_R16G16B16A16_SNORM, 0x09, 0x9 },
   { PIPE_FORMAT_R16G16B16A16_SINT, 0x0a, 0x9 },
   { PIPE_FORMAT_R16G16B16A16_UINT, 0x0b, 0x9 },
   { PIPE_FORMAT_B8G8R8A8_UNORM, 0x11, 0xa },
   { PIPE_FORMAT_R8G8B8A8_UNORM, 0x18, 0xa },
   { PIPE_FORMAT_R8G8B8A8_SNORM, 0x1a, 0xa },
   { PIPE_FORMAT_R8G8B8A8_SINT, 0x1b, 0xa },
   { PIPE_FORMAT_R8G8B8A8_UINT, 0x1c, 0xa },
   { PIPE_FORMAT_R11G11B10_FLOAT, 0x24, 0xa },
   { PIPE_FORMAT_R10G10B10A2_UNORM, 0x13, 0xa },
   { PIPE_FORMAT_R10G10B10A2_UINT, 0x15, 0xa },
   { PIPE_FORMAT_R32G32_FLOAT, 0x0d, 0x4 },
   { PIPE_FORMAT_R32G32_SINT, 0x0e, 0x4 },
   { PIPE_FORMAT_R32G32_UINT, 0x0f, 0x4 },
   { PIPE_FORMAT_R16G16_FLOAT, 0x21, 0x5 },
   { PIPE_FORMAT_R16G16_UNORM, 0x1d, 0x5 },
   { PIPE_FORMAT_R16G16_SNORM, 0x1e, 0x5 },
   { PIPE_FORMAT_R16G16_SINT, 0x1f, 0x5 },
   { PIPE_FORMAT_R16G16_UINT, 0x20, 0x5 },
   { PIPE_FORMAT_R8G8_UNORM, 0x2e, 0x6 },
   { PIPE_FORMAT_R8G8_SNORM, 0x2f, 0x6 },
   { PIPE_FORMAT_R8G8_SINT, 0x30, 0x6 },
   { PIPE_FORMAT_R8G8_UINT, 0x31, 0x6 },
   { PIPE_FORMAT_R32_FLOAT, 0x29, 0x0 },
   { PIPE_FORMAT_R32_SINT, 0x27, 0x0 },
   { PIPE_FORMAT_R32_UINT, 0x28, 0x0 },
   { PIPE_FORMAT_R16_FLOAT, 0x36, 0x1 },
   { PIPE_FORMAT_R16_UNORM, 0x32, 0x1 },
   { PIPE_FORMAT_R16_SNORM, 0x33, 0x1 },
   { PIPE_FORMAT_R16_SINT, 0x34, 0x1 },
   { PIPE_FORMAT_R16_UINT, 0x35, 0x1 },
   { PIPE_FORMAT_R8_UNORM, 0x37, 0x2 },
   { PIPE_FORMAT_R8_SNORM, 0x38, 0x2 },
   { PIPE_FORMAT_R8_SINT, 0x39, 0x2 },
   { PIPE_FORMAT_R8_UINT, 0x3a, 0x2 },
};

/* Texels as memory holds them, the first byte first. */
static const uint8_t texels[][16] = {
   { 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0,
     0x00, 0x00, 0x20, 0x41, 0xcd, 0xcc, 0x4c, 0x3e },
   { 0x00, 0x3c, 0x00, 0xc0, 0xff, 0x7b, 0x01, 0x00,
     0x00, 0x7c, 0x66, 0x36, 0x00, 0x80, 0x55, 0xd5 },
   { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
     0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
   { 0x80, 0x7f, 0x81, 0x01, 0x00, 0x80, 0xff, 0x7f,
     0x01, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80 },
   { 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
     0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78 },
   { 0 },
};

struct device {
   /* The descriptor's `format_info`. */
   uint32_t format_info;
   const uint8_t *texel;
   unsigned raw_loads;
};

/* What a Kepler GPU and NVK's descriptor lowering give the lowered load:
 * the descriptor's words, of which only the format decides anything once
 * the address is gone, and the texel's first bits, as many as each raw load
 * asks for. Bits a load of fewer than 32 does not ask for are set, since
 * the hardware promises nothing of them.
 */
static bool
stand_in_device(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   struct device *device = data;
   b->cursor = nir_before_instr(&intrin->instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_image_deref_load_info_nv: {
      const bool format =
         nir_intrinsic_base(intrin) == offsetof(struct nil_su_info, format_info);
      CHECK(intrin->def.num_components == 1);
      nir_def_replace(&intrin->def,
                      nir_imm_int(b, format ? device->format_info : 0));
      return true;
   }
   case nir_intrinsic_suldga_nv: {
      const unsigned bits =
         util_format_get_blocksizebits(nir_intrinsic_format(intrin));
      CHECK(bits == 8 || bits == 16 || bits == 32 || bits == 64 || bits == 128);
      CHECK(intrin->def.num_components == DIV_ROUND_UP(bits, 32));
      nir_const_value words[4] = { 0 };
      for (unsigned i = 0; i < intrin->def.num_components; i++) {
         uint32_t word;
         memcpy(&word, device->texel + 4 * i, 4);
         if (bits < 32)
            word = (word & BITFIELD_MASK(bits)) | ~BITFIELD_MASK(bits);
         words[i] = nir_const_value_for_uint(word, 32);
      }
      nir_def_replace(&intrin->def,
                      nir_build_imm(b, intrin->def.num_components, 32, words));
      device->raw_loads++;
      return true;
   }
   default:
      return false;
   }
}

static bool
find_stored(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   (void)b;
   if (intrin->intrinsic != nir_intrinsic_store_ssbo)
      return false;
   *(nir_intrinsic_instr **)data = intrin;
   return false;
}

/* The four words a load of `texel` gives a shader, lowered for Kepler with
 * `load_format` on the load (none for a load without a format) from an
 * image whose view has `view`'s format. `raw_loads` takes how many raw
 * texel loads the lowered shader holds.
 */
static void
load(const struct nak_compiler *compiler, enum pipe_format load_format,
     const struct format *view, const uint8_t *texel, uint32_t rgba[4],
     unsigned *raw_loads)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  nak_nir_options(compiler),
                                                  "formatless load");
   const bool integer = util_format_is_pure_integer(view->format);
   const struct glsl_type *type =
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false,
                      integer ? GLSL_TYPE_UINT : GLSL_TYPE_FLOAT);
   nir_variable *image = nir_variable_create(b.shader, nir_var_image, type, "image");
   image->data.image.format = PIPE_FORMAT_NONE;
   nir_deref_instr *deref = nir_build_deref_var(&b, image);
   nir_def *texel_value =
      nir_image_deref_load(&b, 4, 32, &deref->def, nir_imm_ivec4(&b, 3, 5, 0, 0),
                           nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                           .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
                           .format = load_format,
                           .dest_type = integer ? nir_type_uint32 : nir_type_float32);
   nir_store_ssbo(&b, texel_value, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = 0xf, .align_mul = 4);

   CHECK(nak_nir_lower_image_addrs(b.shader, compiler));
   nir_validate_shader(b.shader, "after lowering a Kepler image load");

   struct device device = {
      .format_info = view->hw | (uint32_t)view->description << 8 | 0x4 << 12,
      .texel = texel,
   };
   CHECK(nir_shader_intrinsics_pass(b.shader, stand_in_device,
                                    nir_metadata_none, &device));
   nir_validate_shader(b.shader, "after standing in for the device");
   *raw_loads = device.raw_loads;

   /* Folding alone, with no algebraic rewriting: a rewrite made before a
    * value is known constant (a division turned into a multiplication by
    * the reciprocal, as NAK's options ask) folds to another last bit than
    * the operation written, and where the constant becomes known differs
    * between the two shaders. On a device neither is constant and both run
    * the same operations.
    */
   bool progress;
   unsigned rounds = 0;
   do {
      progress = false;
      NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      NIR_PASS(progress, b.shader, nir_opt_constant_folding);
      NIR_PASS(progress, b.shader, nir_opt_dead_cf);
      NIR_PASS(progress, b.shader, nir_opt_remove_phis);
      NIR_PASS(progress, b.shader, nir_opt_dce);
      CHECK(++rounds < 64);
   } while (progress);

   nir_intrinsic_instr *stored = NULL;
   nir_shader_intrinsics_pass(b.shader, find_stored, nir_metadata_all, &stored);
   CHECK(stored);
   if (!nir_src_is_const(stored->src[0])) {
      fprintf(stderr, "%s read %s a format does not fold to a constant:\n",
              util_format_short_name(view->format),
              load_format == PIPE_FORMAT_NONE ? "without" : "with");
      nir_print_shader(b.shader, stderr);
      exit(1);
   }
   for (unsigned i = 0; i < 4; i++)
      rgba[i] = (uint32_t)nir_src_comp_as_uint(stored->src[0], i);
   ralloc_free(b.shader);
}

/* NVK's descriptor lowering reads the descriptor's words from memory. */
static bool
stand_in_descriptor(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_image_deref_load_info_nv)
      return false;
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *word =
      nir_load_global_constant(b, 1, 32,
                               nir_imm_int64(b, 0x200000 + nir_intrinsic_base(intrin)),
                               .align_mul = 4);
   nir_def_replace(&intrin->def, word);
   return true;
}

/* The lowered load compiles through NAK for Kepler: its raw loads, its
 * branches on the texel's size and its unpacking are instructions the
 * shader model has. Returns the instruction count.
 */
static unsigned
compiles(const struct nak_compiler *compiler, enum pipe_format load_format)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  nak_nir_options(compiler),
                                                  "compiled load");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   const struct glsl_type *type =
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT);
   nir_variable *image = nir_variable_create(b.shader, nir_var_image, type, "image");
   image->data.image.format = PIPE_FORMAT_NONE;
   nir_deref_instr *deref = nir_build_deref_var(&b, image);
   /* A position the compiler does not know. */
   nir_def *at = nir_load_global_constant(&b, 2, 32, nir_imm_int64(&b, 0x100000),
                                          .align_mul = 4);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, at, 0), nir_channel(&b, at, 1),
                             nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_def *texel =
      nir_image_deref_load(&b, 4, 32, &deref->def, coord, nir_imm_int(&b, 0),
                           nir_imm_int(&b, 0),
                           .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
                           .format = load_format, .dest_type = nir_type_float32);
   nir_store_global(&b, texel, nir_imm_int64(&b, 0x300000), .write_mask = 0xf,
                    .align_mul = 4);

   CHECK(nak_nir_lower_image_addrs(b.shader, compiler));
   CHECK(nir_shader_intrinsics_pass(b.shader, stand_in_descriptor,
                                    nir_metadata_control_flow, NULL));
   nir_opt_dce(b.shader);
   nir_remove_dead_derefs(b.shader);
   nir_remove_dead_variables(b.shader, nir_var_image, NULL);
   const nir_lower_compute_system_values_options system_values = {0};
   nir_lower_compute_system_values(b.shader, &system_values);
   nir_validate_shader(b.shader, "before compiling a Kepler image load");
   nak_preprocess_nir(b.shader, compiler);
   struct nak_shader_bin *bin =
      nak_compile_shader(b.shader, false, compiler, 0, NULL, false);
   CHECK(bin && bin->code && bin->code_size &&
         bin->info.stage == MESA_SHADER_COMPUTE);
   const unsigned instructions = bin->info.num_instrs;
   nak_shader_bin_destroy(bin);
   ralloc_free(b.shader);
   return instructions;
}

static uint32_t
bits_of(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, 4);
   return bits;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   /* GK104 to GK107: the Kepler A class at shader model 30. */
   struct nv_device_info info = {
      .type = NV_DEVICE_TYPE_DIS,
      .cls_eng3d = KEPLER_A,
      .sm = 30,
      .max_warps_per_mp = 64,
   };
   struct nak_compiler *compiler = nak_compiler_create(&info);
   CHECK(compiler);

   unsigned checked = 0;
   for (unsigned f = 0; f < ARRAY_SIZE(formats); f++) {
      const struct format *format = &formats[f];
      for (unsigned t = 0; t < ARRAY_SIZE(texels); t++) {
         uint32_t with[4], without[4];
         unsigned loads_with, loads_without;
         load(compiler, format->format, format, texels[t], with, &loads_with);
         load(compiler, PIPE_FORMAT_NONE, format, texels[t], without,
              &loads_without);
         /* One raw load with the format, one of each texel size without. */
         CHECK(loads_with == 1 && loads_without == 5);
         if (memcmp(with, without, sizeof(with))) {
            fprintf(stderr,
                    "%s texel %u: with its format %08x %08x %08x %08x, "
                    "without %08x %08x %08x %08x\n",
                    util_format_short_name(format->format), t,
                    with[0], with[1], with[2], with[3],
                    without[0], without[1], without[2], without[3]);
            exit(1);
         }
         checked++;
      }
   }

   /* Values written out: 8-bit normalized components in memory order with
    * the blue and red of BGRA exchanged, and one 32-bit float with the
    * components its format lacks.
    */
   uint32_t rgba[4];
   unsigned loads;
   const struct format bgra8 = { PIPE_FORMAT_B8G8R8A8_UNORM, 0x11, 0xa };
   load(compiler, PIPE_FORMAT_NONE, &bgra8, texels[4], rgba, &loads);
   CHECK(rgba[0] == bits_of(0x56 / 255.0f) && rgba[1] == bits_of(0x34 / 255.0f) &&
         rgba[2] == bits_of(0x12 / 255.0f) && rgba[3] == bits_of(0x78 / 255.0f));
   const struct format r32f = { PIPE_FORMAT_R32_FLOAT, 0x29, 0x0 };
   load(compiler, PIPE_FORMAT_NONE, &r32f, texels[0], rgba, &loads);
   CHECK(rgba[0] == bits_of(1.0f) && rgba[1] == 0 && rgba[2] == 0 &&
         rgba[3] == bits_of(1.0f));
   const struct format rg16ui = { PIPE_FORMAT_R16G16_UINT, 0x20, 0x5 };
   load(compiler, PIPE_FORMAT_NONE, &rg16ui, texels[4], rgba, &loads);
   CHECK(rgba[0] == 0x3412 && rgba[1] == 0x7856 && rgba[2] == 0 && rgba[3] == 1);
   const struct format r8sn = { PIPE_FORMAT_R8_SNORM, 0x38, 0x2 };
   load(compiler, PIPE_FORMAT_NONE, &r8sn, texels[3], rgba, &loads);
   CHECK(rgba[0] == bits_of(-1.0f) && rgba[1] == 0 && rgba[2] == 0 &&
         rgba[3] == bits_of(1.0f));

   printf("%u Kepler storage reads without a format unpack as with it\n", checked);

   const unsigned with_format = compiles(compiler, PIPE_FORMAT_R8G8B8A8_UNORM);
   const unsigned without_format = compiles(compiler, PIPE_FORMAT_NONE);
   CHECK(with_format && without_format > with_format);
   printf("SM30 compiles the read in %u instructions with its format and %u without\n",
          with_format, without_format);
   nak_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
