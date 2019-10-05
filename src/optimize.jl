f(theta)               = simloglik!(grad, info_matrix, tmpvars, model, theta, data, false)
g!(grad, theta)        = simloglik!(grad, info_matrix, tmpvars, model, theta, data, true)
h!(info_matrix, theta) = simloglik!(grad, info_matrix, tmpvars, model, theta, data, true)
fg! = g!
newton!(f, g, h, theta, x0)
