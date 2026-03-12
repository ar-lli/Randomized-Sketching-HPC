function [x] = fwht(k, x)

if k == 0
    return;
end

k1 = 2^(k-1);
p1 = fwht(k-1, x(1:k1));
p2 = fwht(k-1, x(k1+1:end));

x = [ p1 + p2 ; p1 - p2 ];

end