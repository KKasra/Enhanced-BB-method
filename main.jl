using ForwardDiff
include("algorithm_3_1.jl")
include("unconstrainted_test_problems.jl")

alpha_max=1e6
alpha_min=1e-6
M = 10
sigma=1e-4
delta = .5
tau_1 = .2
gamma = 1.02
epsilon = 1e-6



#testing algorithm(3.1) on quadratic programs
function main()
    n = 100
    # loss, A,q = generate_quadratic_program(n)
    # print('!')
    x_0 = rand(n) * 100
    # loss, x_0, optim = get_Powell_badly_scaled_probem()
    loss, x_0, optim = get_Rosenbrock_problem()
    alpha_2 = alpha_1 = 1/norm(ForwardDiff.gradient(loss, x_0), Inf)

    default_paramters=( alpha_max, alpha_min, tau_1, gamma, epsilon, sigma, delta,M, alpha_1, alpha_2)

    x_sequence, f_sequence = optimize_with_algorithm_3_1(loss,x_0, default_paramters)

    return x_sequence, optim, f_sequence
end