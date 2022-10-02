
function writexyz(io, atoms, r)
    b2Å = 1.0 / 1.8897261245650618
    println(io, length(atoms), '\n')
    for (a, (x, y, z)) in zip(atoms, eachcol(r))
        println(io, a, "    ", x * b2Å, " ", y * b2Å, " ", z * b2Å)
    end
end
