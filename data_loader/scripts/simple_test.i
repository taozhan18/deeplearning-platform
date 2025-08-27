[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = {{nx}}
[]

[Variables]
  [u]
  []
[]

[Kernels]
  [diff]
    type = Diffusion
    variable = u
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = {{left_bc}}
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = right
    value = {{right_bc}}
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
[]

[Outputs]
  exodus = true
  csv = true
  file_base = output
[]