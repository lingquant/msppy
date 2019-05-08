This readme explains how to replicate the four implementations in the paper.

## Hydro-thermal power system planning
Go to the examples/hydro_thermal directory. Run the replicate.sh script. This should generate Table 2, Figure 1, Table 3 and Table 4 in the result folder.

## Hydro-thermal power system planning with integers
Go to the examples/hydro_thermal directory. Run the replicate_TS_integer.sh script. This should produce Table 5 in the console. The third line may not be recovered exactly the same since the stopping criteria is a time limit (1000 seconds).

## Multi-period portfolio optimization
Go to the examples/portfolio_optimization directory. Run the replicate.sh script. This should generate Table 6, Figure 2, Table 7 and Table 8.

## Airline-revenue management
Go to the examples/airline_revenue_management directory. Run the replicate.sh script. This should produce Table 9, Figure 3, Table 10 and Figure 4.  

## Table 11
Go to the examples/hydro_thermal directory. Run the replicate_comparison.sh. This should produce the last half of table 11. To replicate the first half of table, install the SDDP.jl (see https://odow.github.io/SDDP.jl/latest/ and install the old version with Julia 0.6). Go to the examples/hydro_thermal/julia directory. Run the replicate_comparison.sh.
