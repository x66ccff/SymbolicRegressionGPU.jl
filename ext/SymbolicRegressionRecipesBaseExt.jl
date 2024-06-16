module SymbolicRegressionRecipesBaseExt

using RecipesBase: @recipe, @series, plot
using DynamicExpressions: Node, string_tree
using SymbolicRegression.CoreModule: Options
using SymbolicRegression.HallOfFameModule: HallOfFame, format_hall_of_fame
using SymbolicRegression.MLJInterfaceModule: SRFitResult, SRRegressor
using SymbolicRegression.LoggingModule: convex_hull

import SymbolicRegression.LoggingModule: add_plot_to_log!

function add_plot_to_log!(
    log::Dict; trees, losses, complexities, options, variable_names, log_step, ropt
)
    if ropt.log_every_n.plots > 0 && log_step % ropt.log_every_n.plots == 0
        log["plot"] = plot(
            trees, losses, complexities, options; variable_names=variable_names
        )
    else
        nothing
    end
    return nothing
end

function default_sr_plot end

@recipe function default_sr_plot(fitresult::SRFitResult{<:SRRegressor})
    return fitresult.state[2], fitresult.options
end

# TODO: Add variable names
@recipe function default_sr_plot(hall_of_fame::HallOfFame, options::Options)
    out = format_hall_of_fame(hall_of_fame, options)
    return (out.trees, out.losses, out.complexities, options)
end

@recipe function default_sr_plot(
    trees::Vector{N}, losses::Vector{L}, complexities::Vector{Int}, options::Options
) where {T,L,N<:Node{T}}
    tree_strings = [string_tree(tree, options) for tree in trees]
    log_losses = @. log10(losses + eps(L))
    log_complexities = @. log10(complexities)
    # Add an upper right corner to this for the convex hull calculation:
    push!(log_losses, maximum(log_losses))
    push!(log_complexities, maximum(log_complexities))

    xy = cat(log_complexities, log_losses; dims=2)
    log_hull = convex_hull(xy)

    # Add the first point again to close the hull:
    push!(log_hull, log_hull[1])

    # Then remove the first two points for visualization
    log_hull = log_hull[3:end]

    hull = [10 .^ row for row in log_hull]

    xlabel --> "Complexity"
    ylabel --> "Loss"

    xlims --> (0.5, options.maxsize + 1)

    xscale --> :log10
    yscale --> :log10

    # Main complexity/loss plot:
    @series begin
        label --> "Pareto Front"

        complexities, losses
    end

    # Add on a convex hull:
    @series begin
        label --> "Convex Hull"
        color --> :lightgray

        first.(hull), last.(hull)
    end
end

end
