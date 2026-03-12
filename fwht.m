function y = fwht(x)
    n = length(x);
    h = 1;
    while h < n
        for i = 1:2*h:n
            for j = i:i+h-1
                a = x(j);
                b = x(j+h);
                x(j) = a + b;
                x(j+h) = a - b;
            endfor
        endfor
        h = 2*h;
    endwhile
    y = x / sqrt(n);
endfunction