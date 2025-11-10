function d=sqdist(a,b)

aa = sum(a.^2,1);
bb = sum(b.^2,1);
ab = a'*b; 
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

