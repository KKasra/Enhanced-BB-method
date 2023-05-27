using ForwardDiff
using LinearAlgebra
using Debugger
# using linalg
include("stepsizes.jl")




function optimize_with_algorithm_3_1(loss, x_0, parameters)
    grad(x) = ForwardDiff.gradient(loss, x)
    alpha_max, alpha_min, tau_1, gamma, epsilon, sigma, delta,M, alpha_1, alpha_2 = parameters
    
    k = 1
    x_history = [x_0]
    g_history = [grad(x_0)]
    f_history = [loss(x_0)]
    alpha_history = [alpha_1, alpha_2]
    tau = tau_1
    while norm(g_history[k], Inf) > epsilon
        # println(x_history)
        if true || k % 100 == 0
            println(f_history[end])
        end
        descent_direction = - g_history[k]
        lmbda = 0;lmbda = alpha_history[k]
        
        
        #preform GLL
        f_r = maximum(f_history[k-min(M-1,k-1):k])
        
        while loss(x_history[k] + lmbda * descent_direction) >= f_r + sigma * lmbda * g_history[k]' * descent_direction
            
            lmbda *= delta
        end

        new_x = x_history[k] + lmbda * descent_direction

        push!(x_history,new_x)
        push!(g_history, grad(new_x))
        push!(f_history, loss(new_x))

        s_k = x_history[k+1] - x_history[k]
        y_k = g_history[k+1] - g_history[k]

        if k >= 2
            new_alpha = 0
            if s_k' * y_k > 0
                s_k_1 = x_history[k] - x_history[k-1]
                y_k_1 = g_history[k] - g_history[k-1]
                
                if BB2(s_k_1, y_k_1)/BB1(s_k_1, y_k_1)< tau && s_k_1'*y_k_1 > 0
                    if new(s_k, s_k_1,y_k,y_k_1) > 0
                        new_alpha = min(min(BB2(s_k_1,y_k_1), BB2(s_k,y_k)),new(s_k, s_k_1,y_k,y_k_1))
                    else
                        new_alpha = min(BB2(s_k_1,y_k_1), BB2(s_k,y_k))
                    end
                    tau *= 1/gamma
                else
                    new_alpha = BB1(s_k,y_k)
                    tau *= gamma
                end
            else
                new_alpha = min(1/norm(g_history[k],Inf), norm(x_history[k],Inf)/norm(g_history[k],Inf))
            end
            # chop extreme values
            new_alpha = max(new_alpha, alpha_min)
            new_alpha = min(new_alpha, alpha_max)
            append!(alpha_history, new_alpha)
        end
        k += 1
    end

    return x_history, f_history
end

