with open('src/types/feature.rs', 'r') as f:
    content = f.read()

target = "            Self::HasDuplicate => C::HAS_DUPLICATE | C::NEEDS_SORT,"
rep = "            Self::HasDuplicate => C::HAS_DUPLICATE | C::NEEDS_SORT,\n            Self::PkPkDistance => C::MIN | C::MAX,\n            Self::ZeroCross => C::ZERO_CROSS,\n            Self::MaxPowerSpectrum => C::ANY_FFT,"

if "Self::PkPkDistance" not in content:
    content = content.replace(target, rep)
    with open('src/types/feature.rs', 'w') as f:
        f.write(content)

with open('src/types/parse.rs', 'r') as f:
    content = f.read()

target2 = '            Feature::HasDuplicate => "has_duplicate".to_string(),'
rep2 = '            Feature::HasDuplicate => "has_duplicate".to_string(),\n            Feature::PkPkDistance => "pk_pk_distance".to_string(),\n            Feature::ZeroCross => "zero_cross".to_string(),\n            Feature::MaxPowerSpectrum => "max_power_spectrum".to_string(),'

if "Feature::PkPkDistance" not in content[content.find('impl Feature {'):]:
    content = content.replace(target2, rep2)
    with open('src/types/parse.rs', 'w') as f:
        f.write(content)
