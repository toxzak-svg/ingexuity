using Pkg
pkgs = [
    "ONNX",
    "safetensors"
]

for pkg_name in pkgs
    try
        pkg_spec = PackageSpec(name=pkg_name)
        deps = dependent_packages(pkg_spec)
        println("$pkg_name: ", [d.name for d in deps])
    catch e
        println("$pkg_name: not found or error - $e")
    end
end