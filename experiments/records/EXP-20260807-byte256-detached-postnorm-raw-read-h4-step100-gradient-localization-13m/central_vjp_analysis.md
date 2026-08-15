# Central operation VJP amplification

Checkpoint step: 100; batch: 8; forward max error: 0.0

Local L2 gain is `input adjoint L2 / output adjoint L2` for the same executed operation. Branch-alignment rows are kept separate.

| loss | step | operation | local L2 gain | local RMS gain |
|---|---:|---|---:|---:|
| H2 | 1 | rank1_write | 5.67652250 | 18.44056710 |
| H3 | 1 | rank1_write | 5.57767147 | 18.11944286 |
| H4 | 1 | rank1_write | 5.55728666 | 18.05322141 |
| mean_h1_h4 | 1 | rank1_write | 4.95946232 | 16.11114865 |
| H1 | 1 | rank1_write | 4.65573106 | 15.12445710 |
| H2 | 2 | rank1_write | 4.18201700 | 13.58556497 |
| mean_h1_h4 | 2 | rank1_write | 4.13899897 | 13.44581798 |
| H3 | 2 | rank1_write | 4.13691741 | 13.43905589 |
| H4 | 2 | rank1_write | 4.10878477 | 13.34766510 |
| H4 | 3 | complex_read_combined_inputs | 3.46221356 | 0.85192178 |
| H3 | 3 | rank1_write | 3.42175339 | 11.11579723 |
| mean_h1_h4 | 3 | rank1_write | 3.40671107 | 11.06693122 |
| H4 | 3 | rank1_write | 3.39732141 | 11.03642829 |
| H4 | 4 | rank1_write | 3.31535170 | 10.77014412 |
| mean_h1_h4 | 4 | rank1_write | 3.31535170 | 10.77014412 |
| H4 | 4 | complex_read_combined_inputs | 3.28442204 | 0.80817391 |
| mean_h1_h4 | 4 | complex_read_combined_inputs | 3.28442204 | 0.80817391 |
| H4 | 3 | complex_read_query_edge | 3.25609984 | 4.53229916 |
| H4 | 4 | complex_read_query_edge | 3.06722670 | 4.26939888 |
| mean_h1_h4 | 4 | complex_read_query_edge | 3.06722670 | 4.26939888 |
| H4 | 2 | complex_read_combined_inputs | 2.74499286 | 0.67544048 |
| H3 | 2 | complex_read_combined_inputs | 2.74433762 | 0.67527925 |
| mean_h1_h4 | 3 | complex_read_combined_inputs | 2.72496946 | 0.67051347 |
| H3 | 3 | complex_read_combined_inputs | 2.72461594 | 0.67042648 |
| mean_h1_h4 | 3 | complex_read_query_edge | 2.45706549 | 3.42009041 |
| H3 | 3 | complex_read_query_edge | 2.45663720 | 3.41949427 |
| H4 | 2 | complex_read_query_edge | 2.44381736 | 3.40164980 |
| H3 | 2 | complex_read_query_edge | 2.44311449 | 3.40067144 |
| mean_h1_h4 | 2 | complex_read_combined_inputs | 2.22884651 | 0.54843609 |
| H2 | 2 | complex_read_combined_inputs | 2.22772313 | 0.54815967 |
| H2 | 1 | complex_read_combined_inputs | 2.04360661 | 0.50285545 |
| H3 | 1 | complex_read_combined_inputs | 2.01893072 | 0.49678364 |
| H4 | 1 | complex_read_combined_inputs | 2.00561899 | 0.49350812 |
| mean_h1_h4 | 2 | complex_read_query_edge | 1.84496349 | 2.56808050 |
| H2 | 2 | complex_read_query_edge | 1.84351941 | 2.56607042 |
| H2 | 1 | complex_read_query_edge | 1.72935558 | 2.40716109 |
| H3 | 1 | complex_read_query_edge | 1.69482710 | 2.35909948 |
| H4 | 1 | complex_read_query_edge | 1.67805870 | 2.33575885 |
| mean_h1_h4 | 1 | complex_read_combined_inputs | 1.65814241 | 0.40800707 |
| H1 | 1 | complex_read_combined_inputs | 1.65729767 | 0.40779922 |
| H2 | 2 | complex_read_memory_edge | 1.25067443 | 0.31266861 |
| mean_h1_h4 | 2 | complex_read_memory_edge | 1.25054647 | 0.31263662 |
| H4 | 2 | complex_read_memory_edge | 1.25009701 | 0.31252425 |
| H3 | 2 | complex_read_memory_edge | 1.25003222 | 0.31250806 |
| mean_h1_h4 | 1 | complex_read_query_edge | 1.24591054 | 1.73423408 |
| H1 | 1 | complex_read_query_edge | 1.24480109 | 1.73268978 |
| H3 | 3 | complex_read_memory_edge | 1.17833174 | 0.29458294 |
| mean_h1_h4 | 3 | complex_read_memory_edge | 1.17825622 | 0.29456406 |
| H4 | 3 | complex_read_memory_edge | 1.17674829 | 0.29418707 |
| H4 | 4 | complex_read_memory_edge | 1.17454181 | 0.29363545 |
| mean_h1_h4 | 4 | complex_read_memory_edge | 1.17454181 | 0.29363545 |
| H4 | 1 | complex_read_memory_edge | 1.09846554 | 0.27461639 |
| H3 | 1 | complex_read_memory_edge | 1.09710636 | 0.27427659 |
| mean_h1_h4 | 1 | complex_read_memory_edge | 1.09414037 | 0.27353509 |
| H1 | 1 | complex_read_memory_edge | 1.09412331 | 0.27353083 |
| H2 | 1 | complex_read_memory_edge | 1.08887891 | 0.27221973 |
| mean_h1_h4 | 1 | memory_unitary_rotation | 1.00000011 | 1.00000011 |
| H3 | 1 | complex_to_real_flatten | 1.00000011 | 1.00000011 |
| mean_h1_h4 | 3 | memory_unitary_rotation | 1.00000007 | 1.00000007 |
| H1 | 1 | hidden_unitary_rotation | 1.00000000 | 1.00000000 |
| H1 | 1 | memory_unitary_rotation | 1.00000000 | 1.00000000 |
| H1 | 1 | memory_add_rotated_edge | 1.00000000 | 1.00000000 |
| H1 | 1 | memory_add_write_edge | 1.00000000 | 1.00000000 |
| H1 | 1 | complex_to_real_flatten | 1.00000000 | 1.00000000 |
| H1 | 1 | residual_rotated_edge | 1.00000000 | 1.00000000 |
| H1 | 1 | residual_delta_edge | 1.00000000 | 1.00000000 |
| H2 | 1 | hidden_unitary_rotation | 1.00000000 | 1.00000000 |
| H2 | 1 | memory_unitary_rotation | 1.00000000 | 1.00000000 |
| H2 | 1 | memory_add_rotated_edge | 1.00000000 | 1.00000000 |
| H2 | 1 | memory_add_write_edge | 1.00000000 | 1.00000000 |
| H2 | 1 | complex_to_real_flatten | 1.00000000 | 1.00000000 |
| H2 | 1 | residual_rotated_edge | 1.00000000 | 1.00000000 |
| H2 | 1 | residual_delta_edge | 1.00000000 | 1.00000000 |
| H2 | 2 | hidden_unitary_rotation | 1.00000000 | 1.00000000 |
| H2 | 2 | memory_unitary_rotation | 1.00000000 | 1.00000000 |
| H2 | 2 | memory_add_rotated_edge | 1.00000000 | 1.00000000 |
| H2 | 2 | memory_add_write_edge | 1.00000000 | 1.00000000 |
| H2 | 2 | complex_to_real_flatten | 1.00000000 | 1.00000000 |
| H2 | 2 | residual_rotated_edge | 1.00000000 | 1.00000000 |
| H2 | 2 | residual_delta_edge | 1.00000000 | 1.00000000 |
