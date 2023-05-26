function BB1(s_k_1::Vector, y_k_1::Vector)
    return (s_k_1' * s_k_1)/(s_k_1'* y_k_1)
end

function BB2(s_k_1::Vector, y_k_1::Vector)
    return (s_k_1' * y_k_1)/(y_k_1'* y_k_1)
end

function steepest_descent(g_k::Vector, Hess_k::Matrix)
    return (g_k' * g_k) / (g_k' * Hess_k * g_k)
end

function __DY(sd_k, sd_k_1)
    term1 = 1/sd_k_1 + 1/sd_k
    term2 = g_k' * g_k / (sd_k_1 ^ 2 * g_k_1' * g_k_1)

    return 2/(term1 + sqrt(term1 ^ 2 + 4 * term2))
end

function DY(g_k::Vector, g_k_1::Vector, Hess_k::Matrix, Hess_k_1::Matrix)
    sd_k = steepest_descent(g_k, Hess_k)
    sd_k_1 = steepest_descent(g_k_1, Hess_k_1)
    return __DY(sd_k, sd_k_1)
end

function __new(bb1_k::Vector, bb2_k::Vector, bb1_k_1::Vector, bb2_k_1::Vector)
    term1 = (bb2_k_1 - bb2_k) / (bb2_k_1 * bb2_k * (bb1_k_1 - bb1_k))
    term2 = (bb1_k_1 * bb2_k_1 - bb1_k * bb2_k) / (bb2_k_1 * bb2_k * (bb1_k_1 - bb1_k))

    return 1 / (term2 + sqrt(term2^2 - 4 * term1))
end

function new(s_k_1::Vector, s_k_2::Vector, y_k_1::Vector, y_k_2::Vector)
    bb1_k = BB1(s_k_1, y_k_1)
    bb2_k = BB2(s_k_1, y_k_1)

    bb1_k_1 = BB1(s_k_2, y_k_2)
    bb2_k_1 = BB2(s_k_2, y_k_2)
    return __new(bb1_k, bb2_k, bb1_k_1, bb2_k_1)
end



