using DifferentialEquations, Lux, Random, Optimization, OptimizationOptimisers, DiffEqFlux, ComponentArrays
using Plots

# Define ion-channel rate functions
alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0))
beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0))
alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0))
beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0))
alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0)
beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0)

# Define the Hodgkin-Huxley system
function HH!(du, u, p, t)
    gK, gNa, gL, EK, ENa, EL, C, I = p
    v, n, m, h = u

    du[1] = (-(gK * (n^4.0) * (v - EK)) - (gNa * (m^3.0) * h * (v - ENa)) - (gL * (v - EL)) + I) / C
    du[2] = (alpha_n(v) * (1.0 - n)) - (beta_n(v) * n)
    du[3] = (alpha_m(v) * (1.0 - m)) - (beta_m(v) * m)
    du[4] = (alpha_h(v) * (1.0 - h)) - (beta_h(v) * h)
end

# Initial parameters, state, and setup
current_step= PresetTimeCallback(50, integrator -> integrator.p[8] += 1)

# n, m & h steady-states
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))

p = [35.0, 40.0, 0.3, -77.0, 55.0, -65.0, 1, 0]
u0 = [-60, n_inf(-60), m_inf(-60), h_inf(-60)]
tspan = (0.0, 200)

# Solve the standard ODE problem
prob = ODEProblem(HH!, u0, tspan, p, callback = current_step)
sol = solve(prob, AutoTsit5(Rosenbrock23()); saveat= 0.1)
plot(sol, vars = 1)

# Prepare data for NeuralODE training
ode_data = Array(sol)

# Neural network for NeuralODE
rng = Random.default_rng()
dudt = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 4))  # 4 inputs, 50 hidden, 4 outputs
p_nn, st_nn = Lux.setup(rng, dudt)

# Convert p_nn to a mutable structure (ComponentArray)
p_nn_mutable = ComponentArray(p_nn)

# NeuralODE setup
neuralode = NeuralODE(dudt, tspan, AutoTsit5(Rosenbrock23()); saveat= sol.t)

# Prediction function
function predict_neuralode(p_nn_mutable)
    neural_sol = neuralode(u0, p_nn_mutable, st_nn)
    Array(neural_sol[1])  # Extract predictions
end

# Loss function
function loss_neuralode(p_nn_mutable)
    pred = predict_neuralode(p_nn_mutable)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

# Training callback
losses = Float64[]
callback = function (p_nn_mutable, l; doplot = true)
    push!(losses, l)
    println("Current loss: $(l)")
    if doplot
        plt = scatter(sol.t, ode_data[1, :], label="Data")
        scatter!(plt, sol.t, predict_neuralode(p_nn_mutable)[1, :], label="Prediction")
        display(plot(plt))
    end
    return false
end

# Optimization using Adam
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_nn_mutable)

# First phase: Adam optimizer
result_adam = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.09); callback=callback, maxiters=10)
println("Training loss after Adam: $(losses[end])")

# Second phase: BFGS optimizer
optprob_bfgs = Optimization.OptimizationProblem(optf, result_adam.u)
result_bfgs = Optimization.solve(optprob_bfgs, Optim.BFGS(; initial_stepnorm = 0.01); callback=callback, maxiters=100)
println("Final training loss after BFGS: $(losses[end])")

# Plot the losses
pl_losses = plot(1:length(losses), losses, yaxis=:log10, xlabel="Iterations", ylabel="Loss",
                 label="Loss", color=:blue)
