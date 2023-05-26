using Random
using LinearAlgebra
function setseed(seed)
    Random.seed!(seed)
end

setseed(0)

function generate_positive_definite_matirx(n)
    lmbda = diagm(rand(Float64, n))
    Q,_ = qr(rand(Float64,(n,n)))
    return Q * lmbda * Q'
end

function generate_quadratic_program(n)
    A = generate_positive_definite_matirx(n)
    q = rand(Float64,n)
    f(x::Vector) = x' * A * x + q' * x
    return f, A, q
end

function get_Rosenbrock_problem();
    f1(x) = 10(x[2] - x[1]^2)
    f2(x) = 1 - x[1]
    
    x_0 = [-1.2, 1]
    optimum = [1,1]

    loss(x) = f1(x) ^2 + f2(x) ^ 2

    return loss, x_0, optimum

end

function get_Powell_badly_scaled_probem()
    f1(x) = 1e4 * x[1] * x[2] - 1
    f2(x) = epx(-x[1]) + exp(-x[2]) - 1.0001
    x_0 = [0,1]
    optimum = [1.098e-5, 9.106]

    loss = f1(x)^2 + f2(x)^2
    return loss, x_0, optimum
end

