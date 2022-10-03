
function get_num_hessian_2(ef, r, h)
    e0 = ef(r)
    r_buf = copy(r)

    hess = zeros(size(r, 2), size(r, 2), size(r, 1), size(r, 1))

    natm = size(r, 2)

    for a1 in 1:natm, a2 in 1:natm, q1 in 1:3, q2 in 1:3
        h_ij = if (a1, q1) == (a2, q2)
            r_buf[q1, a1] = r[q1, a1] + h
            e1 = ef(r_buf)

            r_buf[q1, a1] = r[q1, a1] - h
            e2 = ef(r_buf)

            r_buf[q1, a1] = r[q1, a1]

            (e1 + e2 - 2 * e0) / h^2
        else
            r_buf[q1, a1] = r[q1, a1] + h
            r_buf[q2, a2] = r[q2, a2] + h
            e11 = ef(r_buf)

            r_buf[q2, a2] = r[q2, a2] - h
            e12 = ef(r_buf)

            r_buf[q1, a1] = r[q1, a1] - h
            e22 = ef(r_buf)

            r_buf[q2, a2] = r[q2, a2] + h
            e21 = ef(r_buf)

            r_buf[q1, a1] = r[q1, a1]
            r_buf[q2, a2] = r[q2, a2]

            (e11 + e22 - e12 - e21) / (4 * h^2)
        end

        hess[a1, a2, q1, q2] = h_ij
    end

    hess
end
