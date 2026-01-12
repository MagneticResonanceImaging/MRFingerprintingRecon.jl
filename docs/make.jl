using Pkg
Pkg.activate("docs")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()

using MRISubspaceRecon
using Documenter
using Literate
using Plots # to not capture precompilation output

# HTML Plotting Functionality
struct HTMLPlot
    p # :: Plots.Plot
end
const ROOT_DIR = joinpath(@__DIR__, "build")
const PLOT_DIR = joinpath(ROOT_DIR, "plots")
function Base.show(io::IO, ::MIME"text/html", p::HTMLPlot)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, string(UInt32(floor(rand()*1e9)), ".html"))
    Plots.savefig(p.p, path)
    if get(ENV, "CI", "false") == "true" # for prettyurl
        print(io, "<object type=\"text/html\" data=\"../../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    else
        print(io, "<object type=\"text/html\" data=\"../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    end
end

# Notebook hack to display inline math correctly
function notebook_filter(str)
    re = r"(?<!`)``(?!`)"  # Two backquotes not preceded by nor followed by another
    return replace(str, re => "\$")
end

# Literate
OUTPUT = joinpath(@__DIR__, "src/build_literate")

files = [
    "tutorial.jl",
]

for file in files
    file_path = joinpath(@__DIR__, "src/", file)
    Literate.markdown(file_path, OUTPUT)
    Literate.notebook(file_path, OUTPUT, preprocess=notebook_filter; execute=false)
    Literate.script(  file_path, OUTPUT)
end

DocMeta.setdocmeta!(MRISubspaceRecon, :DocTestSetup, :(using MRISubspaceRecon); recursive=true)

makedocs(;
    doctest = false,
    modules=[MRISubspaceRecon],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo="https://github.com/MagneticResonanceImaging/MRISubspaceRecon.jl/blob/{commit}{path}#{line}",
    sitename="MRISubspaceRecon.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MagneticResonanceImaging.github.io/MRISubspaceRecon.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start Tutorial" => Any[
            "build_literate/tutorial.md",
        ],
        "API" => "api.md",
    ],
)

# Set dark theme as default independent of the OS's settings
run(`sed -i'.old' 's/var darkPreference = false/var darkPreference = true/g' docs/build/assets/themeswap.js`)

deploydocs(;
    repo="github.com/MagneticResonanceImaging/MRISubspaceRecon.jl",
)
