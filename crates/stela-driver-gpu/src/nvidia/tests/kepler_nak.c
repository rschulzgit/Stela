/* SPDX-License-Identifier: GPL-2.0-or-later */
/* What the Metal layer asks of every device and NVK does not report on
 * Kepler compiles through NAK there all the same.
 *
 * NVK reports shaderFloat16 from SM70 and vulkanMemoryModel from Maxwell.
 * Both are promises about SPIR-V. The Metal layer hands NVK NIR, so what
 * counts is whether NAK compiles the NIR the layer emits: 16-bit float
 * values with their arithmetic and conversions, and barriers with a memory
 * scope beside atomics on global memory. The layer's requirements waive the
 * two features for NVK on the strength of this (`requirements` in
 * `metal/vulkan/backend.rs`). No device is opened and nothing is run: this
 * checks that the Kepler shader model has code for each, not what the code
 * computes.
 */
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "nouveau/compiler/nak.h"
#include "nouveau/headers/nv_device_info.h"
#include "nouveau/headers/nvidia/classes/cla097.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { if (!(condition)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

static nir_builder
kernel(const struct nak_compiler *compiler, const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  nak_nir_options(compiler), "%s", name);
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static unsigned
compile(nir_builder *b, const struct nak_compiler *compiler, const char *name)
{
   const nir_lower_compute_system_values_options system_values = {0};
   nir_lower_compute_system_values(b->shader, &system_values);
   nir_validate_shader(b->shader, name);
   nak_preprocess_nir(b->shader, compiler);
   struct nak_shader_bin *bin =
      nak_compile_shader(b->shader, false, compiler, 0, NULL, false);
   CHECK(bin && bin->code && bin->code_size &&
         bin->info.stage == MESA_SHADER_COMPUTE);
   const unsigned instructions = bin->info.num_instrs;
   nak_shader_bin_destroy(bin);
   ralloc_free(b->shader);
   return instructions;
}

/* Metal's `half`: values loaded and stored as 16 bits, their arithmetic,
 * comparisons and rounding, and the conversions to and from 32-bit floats
 * and integers.
 */
static unsigned
half_floats(const struct nak_compiler *compiler)
{
   nir_builder b = kernel(compiler, "half floats");
   nir_def *address = nir_imm_int64(&b, 0x100000);
   nir_def *x = nir_load_global(&b, 1, 16, address, .align_mul = 2);
   nir_def *y = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, address, 2),
                                .align_mul = 2);
   nir_def *wide = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, address, 4),
                                   .align_mul = 4);
   nir_def *value = nir_fadd(&b, x, y);
   value = nir_fmul(&b, value, nir_fsub(&b, x, y));
   value = nir_fdiv(&b, value, nir_fmax(&b, nir_fabs(&b, y), nir_imm_floatN_t(&b, 0.5, 16)));
   value = nir_fadd(&b, value, nir_fsqrt(&b, nir_fabs(&b, x)));
   value = nir_ffma(&b, value, x, y);
   value = nir_fmin(&b, value, nir_ffloor(&b, nir_fneg(&b, y)));
   value = nir_bcsel(&b, nir_flt(&b, x, y), nir_fsat(&b, value), nir_fround_even(&b, value));
   value = nir_fadd(&b, value, nir_fsin(&b, x));
   value = nir_fadd(&b, value, nir_fexp2(&b, nir_flog2(&b, nir_fabs(&b, y))));
   /* Between the widths and to and from integers. */
   value = nir_fadd(&b, value, nir_f2f16(&b, wide));
   nir_def *widened = nir_fmul(&b, nir_f2f32(&b, value), wide);
   nir_def *integer = nir_f2i32(&b, value);
   value = nir_fadd(&b, value, nir_i2f16(&b, integer));
   value = nir_fadd(&b, value, nir_u2f16(&b, nir_f2u32(&b, nir_fabs(&b, x))));
   nir_store_global(&b, value, nir_iadd_imm(&b, address, 8), .align_mul = 2);
   nir_store_global(&b, widened, nir_iadd_imm(&b, address, 12), .align_mul = 4);
   /* Two at once, as Metal's half2. */
   nir_def *pair = nir_load_global(&b, 2, 16, nir_iadd_imm(&b, address, 16),
                                   .align_mul = 2);
   nir_store_global(&b, nir_fmul(&b, pair, nir_vec2(&b, x, y)),
                    nir_iadd_imm(&b, address, 20), .align_mul = 2);
   return compile(&b, compiler, "half floats");
}

/* The barriers the layer emits: a threadgroup barrier that orders memory,
 * a fence past the threadgroup on global memory, and an atomic between.
 */
static unsigned
barriers(const struct nak_compiler *compiler)
{
   nir_builder b = kernel(compiler, "barriers");
   /* A buffer's address, which a shader has from its arguments. */
   nir_def *address = nir_load_global_constant(&b, 1, 64, nir_imm_int64(&b, 0x100000),
                                               .align_mul = 8);
   nir_def *before = nir_global_atomic(&b, 32, address, nir_imm_int(&b, 1),
                                       .atomic_op = nir_atomic_op_iadd);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared | nir_var_mem_global | nir_var_mem_ssbo);
   nir_store_global(&b, before, nir_iadd_imm(&b, address, 4), .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_RELEASE,
               .memory_modes = nir_var_mem_global | nir_var_mem_ssbo);
   nir_def *seen = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, address, 8),
                                   .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQUIRE,
               .memory_modes = nir_var_mem_global | nir_var_mem_ssbo);
   nir_def *swapped = nir_global_atomic_swap(&b, 32, address, seen, before,
                                             .atomic_op = nir_atomic_op_cmpxchg);
   nir_store_global(&b, swapped, nir_iadd_imm(&b, address, 12), .align_mul = 4);
   return compile(&b, compiler, "barriers");
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   /* GK104 to GK107: the Kepler A class at shader model 30. */
   const struct nv_device_info kepler = {
      .type = NV_DEVICE_TYPE_DIS,
      .cls_eng3d = KEPLER_A,
      .sm = 30,
      .max_warps_per_mp = 64,
   };
   struct nak_compiler *compiler = nak_compiler_create(&kepler);
   CHECK(compiler);
   printf("SM30 half floats: %u instructions\n", half_floats(compiler));
   printf("SM30 barriers: %u instructions\n", barriers(compiler));
   nak_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
