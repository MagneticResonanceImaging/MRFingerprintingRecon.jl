
"""
    data_array!(data)

Convert data from Array to vector representation

# Arguments
- `data::Array{ComplexF32}`: Dataset in array representation
"""
function data_vec(data; Cartesian = false)

    if Cartesian
        data = [data[:,:,i] for i=1:size(data,3)]

    else
        if 3 == length(size(data))
            data = [data[:,i,:] for i=1:size(data,2)]

        elseif 4 == length(size(data))
            data = [data[:,i,:,:] for i=1:size(data,2)]
        else
            error("Data type is not supported!")
        end
    end
    return data
end


"""
    data_array!(data)

Convert data from vector to array representation

# Arguments
- `data::Vector{Array{ComplexF32}}`: Dataset in vector representation
"""
function data_array(data; Cartesian = false)

    if Cartesian
        data = combinedimsview(data)

    else
        if 2 == length(size(data[1]))
            data = cat(data..., dims = 3)
            data = permutedims(data, (1,3,2))

        elseif 3 == length(size(data[1]))
            data = cat(data..., dims = 4)
            data = permutedims(data, (1,4,2,3))
        else
            error("Data type is not supported!")
        end
    end
    return data
end