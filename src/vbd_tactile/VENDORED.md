# Vendored solver subset

These 96 modules are copied byte-for-byte from `src/vbd_tactile` of the
`newton-tactile-simulation` project at commit `c012fc3` (master, 2026-11-13). They are the
static import closure of `revo3.Revo3TactileHand`, `rigid_mesh` and `rigid_mesh_ipc`: the
prescribed-motion fingertip-gel / rigid-mesh IPC solver ("old solver"). Some modules belong to code
paths this demo never takes (for example the ABD dynamic-object backend), but they are imported by
the same files, so they are kept unchanged.

Only `__init__.py` differs: the original package root imports Isaac Lab and other adapters that
this demo does not need.

| file | sha256 (first 16 hex) |
|---|---|
| `abd_block_rows.py` | `ac6b738a6cc6846e` |
| `abd_block_scan.py` | `3749fd1369a26416` |
| `abd_body_blocks.py` | `8b332d3c08dc8c98` |
| `abd_compact.py` | `983f9d3d4fcc162b` |
| `abd_convergence.py` | `7bf12aca1e8d627c` |
| `abd_coupled_tangent.py` | `2adc905b20d6f145` |
| `abd_dynamics.py` | `d54a89ee0579aea7` |
| `abd_energy_increment.py` | `b4446f7294ee4478` |
| `abd_fixed_hessian.py` | `b48766d7ce37e501` |
| `abd_friction_iteration.py` | `306b6b52bdd0184f` |
| `abd_fused_operator.py` | `8bb07848511149e0` |
| `abd_gpu.py` | `cd11efba5c4d348b` |
| `abd_graph.py` | `caea526d04903e9d` |
| `abd_guides.py` | `4df86be65ba5ab90` |
| `abd_mapping_precision.py` | `fbb217523a73322a` |
| `abd_material_projection.py` | `31a20efc757cdaf1` |
| `abd_operator.py` | `4eed01b7e71a19a3` |
| `abd_path_policy.py` | `71050b93c7a2aec1` |
| `abd_precondition_dot.py` | `41c079dae3d09b2e` |
| `abd_raw_rows.py` | `638a334e42af7fb3` |
| `abd_residual_report.py` | `834ec537dbfc57c9` |
| `abd_rotation_search.py` | `b43253100b61b129` |
| `abd_schwarz.py` | `b0eb6ba776e240dc` |
| `abd_surface_operator.py` | `b5b884775752bdb0` |
| `abd_temporal.py` | `ee9ff756ac30ed86` |
| `abd_trial_paths.py` | `80917e9e4a61360f` |
| `abd_warp_product.py` | `e4a66e7f49751a1d` |
| `curved_markers.py` | `7d3ced9536527f01` |
| `ipc.py` | `1a027f26c3a5ea5b` |
| `ipc_contact_aggregation.py` | `7d0de726a93d9444` |
| `ipc_contact_local_area.py` | `5aed134b096c29ab` |
| `ipc_contact_softmax.py` | `dc055b4344db1b33` |
| `ipc_contact_spectral.py` | `f8d9c67dd46ee795` |
| `ipc_convergent.py` | `913ad2c58bb94423` |
| `ipc_energy_increment.py` | `cccdfe3213313e89` |
| `ipc_feature_projection.py` | `34c261d0237d239e` |
| `ipc_gpu_candidate_cache.py` | `17ec9ca6ca57f9e5` |
| `ipc_gpu_ccd_pairs.py` | `07837159ac352583` |
| `ipc_gpu_contact_hessian.py` | `d6fe08fc68101739` |
| `ipc_gpu_contacts.py` | `fd35c4e8a827f2fe` |
| `ipc_gpu_control.py` | `cf7180d8cb94a48f` |
| `ipc_gpu_energy.py` | `5ecba4b47ff31719` |
| `ipc_gpu_friction_iteration.py` | `ac852f003580be88` |
| `ipc_gpu_graph.py` | `e4316399464cdcb7` |
| `ipc_gpu_grouped_cg.py` | `e3840131badc218a` |
| `ipc_gpu_hessian.py` | `39717ba686dbf788` |
| `ipc_gpu_kernels.py` | `74243d9d9c49c9da` |
| `ipc_gpu_material.py` | `d272418323baffcd` |
| `ipc_gpu_math.py` | `a4d5b84a6b123005` |
| `ipc_gpu_multilevel.py` | `96bf9dda6fa1d849` |
| `ipc_gpu_nested_cg.py` | `b8b15ee44ff25bf7` |
| `ipc_gpu_operator.py` | `6eca316b2ba4ec30` |
| `ipc_gpu_parallel_search.py` | `44b1e7384eee5624` |
| `ipc_gpu_path_cache.py` | `237ff92e2f63e96a` |
| `ipc_gpu_preconditioner.py` | `03871980668011d1` |
| `ipc_gpu_safe_step.py` | `14c158c39b23ea61` |
| `ipc_gpu_seed.py` | `e31d65c242512571` |
| `ipc_gpu_split_contact.py` | `2013bbac02090d14` |
| `ipc_gpu_swept_cache.py` | `82e98cf8ad08ca3a` |
| `ipc_gpu_warp_psd.py` | `728d14c7f6a4ebfd` |
| `ipc_log_material.py` | `3923f049af2f37d8` |
| `ipc_log_material_blocks.py` | `a3384e4aafe5b254` |
| `ipc_log_material_raw.py` | `ceed93a62ebc16d3` |
| `ipc_precision.py` | `6dbd3d535d5d9c53` |
| `ipc_primitives.py` | `e9754246fd6534fd` |
| `ipc_primitives_warp.py` | `bea2b8b6e4bf27f8` |
| `ipc_self_audit.py` | `06b909ea9d6115e7` |
| `ipc_surface.py` | `20c5136a00ccec8e` |
| `ipc_surface_control.py` | `a60d804b6f69f280` |
| `ipc_surface_gpu.py` | `ebc94cfee21f6f66` |
| `ipc_surface_reference.py` | `ca3572d2c7b46c6a` |
| `ipc_surface_warp.py` | `79c1329f01f0339b` |
| `ipc_unified_gpu.py` | `dd0d122b5f1df4f2` |
| `ipc_vbd.py` | `8826840631608444` |
| `ipc_vbd_kernels.py` | `3628bdb01acd4008` |
| `ipc_volume_barrier.py` | `5088bdb3f1b54812` |
| `ipc_warp.py` | `dbe4c38b01b66bb8` |
| `markers.py` | `453048cab5a5f6d0` |
| `mounted_ipc.py` | `c52f42013ed3977e` |
| `optical_geometry_gpu.py` | `91f73d4fc485451e` |
| `perspective_optical.py` | `ec949c1a989f8c16` |
| `perspective_optical_gpu.py` | `0fcaa2d731db5e53` |
| `revo3.py` | `7038e26e7fcb0ff0` |
| `revo3_material.py` | `f9f92b77f1661e66` |
| `rigid_curved_path.py` | `94dec1dddbd11c17` |
| `rigid_dynamics.py` | `8f75b8446ac94c86` |
| `rigid_dynamics_gpu.py` | `29e2b169afc68e40` |
| `rigid_mesh.py` | `4bd4c4c07019b8ab` |
| `rigid_mesh_gpu.py` | `f49582226170654b` |
| `rigid_mesh_ipc.py` | `0018150fe806e45b` |
| `rigid_mesh_planar.py` | `bd1166d5623d7c40` |
| `rigid_schur.py` | `6e6a1f5cc4ea0124` |
| `rigid_warm_start.py` | `50bd05da75958b4b` |
| `surface_geometry.py` | `08e014783a163d26` |
| `taccel_contact.py` | `fb56677dde680e8e` |
| `types.py` | `4d5a61281d218cb0` |
