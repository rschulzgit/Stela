/* SPDX-License-Identifier: GPL-2.0-or-later */
/* The backend's internal compute kernels compile through NAK for Kepler.
 *
 * `metal_abi_nak.c` compiles them for the shader models from Turing, where
 * images are bindless handles. On Kepler NVK lowers every image access to
 * an address computed from the image's descriptor and a raw load or store
 * (nak_nir_lower_image_addrs), reads without a format included (Stela's
 * patch), so the kernels take that path here: the Vulkan driver's
 * descriptor lowering is stood in for by the descriptor's words read from
 * global memory. No device is opened. A kernel named on the command line
 * is compiled alone.
 */
#include "../metal_abi.h"
#include "nouveau/compiler/nak.h"
#include "nouveau/headers/nv_device_info.h"
#include "nouveau/headers/nvidia/classes/cla097.h"
#include "compiler/glsl_types.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

/* What the Vulkan driver's lowering leaves of the kernel's bindings on
 * Kepler, after its image accesses became addresses.
 */
static bool
stand_in_bindings(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   (void)data;
   b->cursor = nir_before_instr(&intrin->instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_push_constant: {
      nir_def *address = nir_iadd_imm(b, nir_u2u64(b, intrin->src[0].ssa),
                                      0x100000 + nir_intrinsic_base(intrin));
      nir_def *value = nir_load_global_constant(b, intrin->def.num_components,
                                                intrin->def.bit_size, address,
                                                .align_mul = 4);
      nir_def_replace(&intrin->def, value);
      return true;
   }
   case nir_intrinsic_image_deref_load_info_nv: {
      /* The descriptor of the image the access names, 64 bytes each. */
      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
      nir_def *index = deref->deref_type == nir_deref_type_array
                          ? nir_u2u64(b, deref->arr.index.ssa) : nir_imm_int64(b, 0);
      nir_def *address = nir_iadd_imm(b, nir_imul_imm(b, index, 64),
                                      0x200000 + nir_intrinsic_base(intrin));
      nir_def *word = nir_load_global_constant(b, intrin->def.num_components, 32,
                                               address, .align_mul = 4);
      nir_def_replace(&intrin->def, word);
      return true;
   }
   default:
      return false;
   }
}

static const struct kernel {
   const char *name;
   nir_shader *(*build)(const nir_shader_compiler_options *);
} kernels[] = {
   { "accumulate", stela_metal_accumulate_kernel },
   { "select", stela_metal_select_kernel },
   { "resolve draws", stela_metal_resolve_draws_kernel },
   { "resolve dispatches", stela_metal_resolve_dispatches_kernel },
   { "store texels", stela_metal_store_texels_kernel },
   { "load texels", stela_metal_load_texels_kernel },
   { "average samples", stela_metal_average_samples_kernel },
   { "decode etc", stela_metal_decode_etc_kernel },
   { "decode pvrtc", stela_metal_decode_pvrtc_kernel },
   { "decode astc", stela_metal_decode_astc_kernel },
   { "reorder blocks", stela_metal_reorder_blocks_kernel },
};

static void
compiles(const struct kernel *kernel)
{
   /* GK104 to GK107: the Kepler A class at shader model 30. */
   const struct nv_device_info device = {
      .type = NV_DEVICE_TYPE_DIS,
      .cls_eng3d = KEPLER_A,
      .sm = 30,
      .max_warps_per_mp = 64,
   };
   struct nak_compiler *compiler = nak_compiler_create(&device);
   CHECK(compiler);
   nir_shader *nir = kernel->build(nak_nir_options(compiler));
   CHECK(nir);
   nak_nir_lower_image_addrs(nir, compiler);
   nir_shader_intrinsics_pass(nir, stand_in_bindings, nir_metadata_control_flow, NULL);
   nir_opt_dce(nir);
   nir_remove_dead_derefs(nir);
   nir_remove_dead_variables(nir, nir_var_image | nir_var_uniform, NULL);
   /* Dispatches start at workgroup zero, as the driver's do. */
   const nir_lower_compute_system_values_options system_values = {0};
   nir_lower_compute_system_values(nir, &system_values);
   nir_validate_shader(nir, kernel->name);
   nak_preprocess_nir(nir, compiler);
   struct nak_shader_bin *bin = nak_compile_shader(nir, false, compiler, 0, NULL, false);
   CHECK(bin && bin->code && bin->code_size && bin->info.stage == MESA_SHADER_COMPUTE);
   printf("%s SM30: %u instructions\n", kernel->name, bin->info.num_instrs);
   fflush(stdout);
   nak_shader_bin_destroy(bin);
   ralloc_free(nir);
   nak_compiler_destroy(compiler);
}

int
main(int argc, char **argv)
{
   glsl_type_singleton_init_or_ref();
   unsigned compiled = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(kernels); i++) {
      if (argc > 1 && strcmp(argv[1], kernels[i].name))
         continue;
      compiles(&kernels[i]);
      compiled++;
   }
   CHECK(compiled);
   puts("metal abi kernels compile through NAK for Kepler");
   glsl_type_singleton_decref();
   return 0;
}
