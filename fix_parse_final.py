with open('src/types/parse.rs', 'r') as f:
    content = f.read()

target = '            "mean_second_derivative_central" => {'
rep = '            "pk_pk_distance" => return Ok(Feature::PkPkDistance),\n            "zero_cross" | "torque_Zero_crossing_rate" => return Ok(Feature::ZeroCross),\n            "max_power_spectrum" => return Ok(Feature::MaxPowerSpectrum),\n            "mean_second_derivative_central" => {'

if "Ok(Feature::PkPkDistance)" not in content[:content.find('fn parse_parameterized')]:
    content = content.replace(target, rep)
    with open('src/types/parse.rs', 'w') as f:
        f.write(content)
