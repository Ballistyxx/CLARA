# CLARA Comment Syntax Reference

## Overview
CLARA comments provide enhanced control over component sizing and differential pair definitions in SPICE netlists for RL-based analog IC layout.

## Syntax

### Basic Format
```spice
<component_line> ;CLARA <command> <parameters>
```

## Commands

### 1. override-size
Override SPICE parameters and/or specify custom grid dimensions.

**Syntax:**
```spice
;CLARA override-size [L=value] [W=value] [mx=value] [my=value] [m=value] [nf=value]
```

**Examples:**
```spice
* Override only grid dimensions
XM1 D G S B model L=0.5 W=2.0 ;CLARA override-size mx=3 my=2

* Override both SPICE parameters and grid dimensions
XM1 D G S B model L=1.0 W=5.0 ;CLARA override-size L=2.0 W=4.0 mx=4 my=3

* Override multiplier and add custom sizing
XM1 D G S B model L=0.5 W=2.0 m=2 ;CLARA override-size m=5 mx=3 my=2
```

### 2. pair
Define explicit differential pairs for symmetry matching.

**Syntax:**
```spice
;CLARA pair <pair_name>
```

**Examples:**
```spice
* Differential input pair
XM1 D1 G1 S B model L=1.0 W=5.0 ;CLARA pair diff_input
XM2 D2 G2 S B model L=1.0 W=5.0 ;CLARA pair diff_input

* Current mirror pair
XM3 VDD D1 VDD B pmos L=0.5 W=10.0 ;CLARA pair current_mirror
XM4 VDD D2 VDD B pmos L=0.5 W=10.0 ;CLARA pair current_mirror
```

### 3. Combined Commands
Multiple CLARA commands can be combined in a single comment.

**Examples:**
```spice
* Override parameters and define pair
XM1 D G S B model L=0.5 W=2.0 ;CLARA override-size L=1.5 mx=3 my=2 pair input_diff
XM2 D G S B model L=0.5 W=2.0 ;CLARA override-size L=1.5 mx=3 my=2 pair input_diff

* Full feature combination
XM1 D G S B model L=1.0 W=5.0 m=2 ;CLARA override-size L=2.0 W=4.0 mx=3 my=2 m=4 pair diff1
```

## Parameter Details

### Size Parameters
- **L=value**: Override SPICE length parameter
- **W=value**: Override SPICE width parameter  
- **mx=value**: Grid size X dimension (always takes precedence over calculated width)
- **my=value**: Grid size Y dimension (always takes precedence over calculated height)

### Device Parameters
- **m=value**: Override multiplier (number of parallel devices)
- **nf=value**: Override number of fingers

### Pair Parameters
- **pair name**: Pair name for differential matching (plain text, no brackets)

## Precedence Rules

1. **CLARA parameters always take precedence** over original SPICE parameters
2. **mx/my parameters always override** any calculated dimensions from L/W
3. Parameters not specified in CLARA comments fall back to original SPICE values
4. Multiplier expansion preserves all CLARA parameters in expanded components

## Examples by Circuit Type

### Differential Amplifier
```spice
.subckt diff_amp VIN_P VIN_N VOUT VDD VSS
* Input differential pair with custom sizing and explicit pairing
XM1 VOUT1 VIN_P CS VSS nmos L=1.0 W=5.0 ;CLARA override-size L=2.0 W=4.0 mx=3 my=2 pair diff_input
XM2 VOUT2 VIN_N CS VSS nmos L=1.0 W=5.0 ;CLARA override-size L=2.0 W=4.0 mx=3 my=2 pair diff_input

* Current mirror load with pairing
XM3 VOUT1 VOUT1 VDD VDD pmos L=0.5 W=10.0 ;CLARA pair current_mirror  
XM4 VOUT2 VOUT1 VDD VDD pmos L=0.5 W=10.0 ;CLARA pair current_mirror

* Tail current source with custom sizing
XM5 CS BIAS VSS VSS nmos L=2.0 W=20.0 ;CLARA override-size mx=4 my=3
.ends
```

### Current Mirror
```spice
* Reference transistor
XM1 VOUT VOUT VDD VDD pmos L=0.5 W=10.0 m=2 ;CLARA pair mirror_main
* Mirror transistor
XM2 IOUT VOUT VDD VDD pmos L=0.5 W=20.0 m=4 ;CLARA override-size W=20.0 m=4 pair mirror_main
```

## Notes

- Pair names are case-sensitive
- Multiple components can share the same pair name
- Components in the same pair will be matched for symmetry in RL training
- Grid utilization is calculated based on mx/my values
- All CLARA parameters are preserved through multiplier expansion