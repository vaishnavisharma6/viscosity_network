using DelimitedFiles
function check_const(a)
    mean_a = sum(a)/length(a)
    var_a = 0
    for i in eachindex(a)
       var_a += abs(a[i] - mean_a)
    end
    return var_a ≈ 0.0
 end
 
 data = readdlm("all_data.txt")
 
 for i in 1:size(data,1)
    a = data[i,:]
    @show a
    if check_const(a[1:end-1]) == true
       @assert a[end] ≈ 0.0 i
    end
 end