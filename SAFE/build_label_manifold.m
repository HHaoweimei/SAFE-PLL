function Outputs = build_label_manifold(train_data, train_p_target, k,kdtree)


[p,q]=size(train_p_target);
train_data = normr(train_data);

[neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
neighbor = neighbor(:,2:k+1);
options = optimoptions('quadprog','Display', 'off','Algorithm','interior-point-convex' );
W = zeros(p,p);

for i = 1:p
	train_data1 = train_data(neighbor(i,:),:);
	D = repmat(train_data(i,:),k,1)-train_data1;
	DD = D*D';
	lb = sparse(k,1);                         
	ub = ones(k,1);                                   
	Aeq = ub';                                
	beq = 1;                                  
	w = quadprog(2*DD, [], [],[], Aeq, beq, lb, ub,[], options);
	W(i,neighbor(i,:)) = w';
end

M = sparse(p,p);
WT = W';
T =WT*W+ W*ones(p,p)*WT.*eye(p,p)-2*WT;     
T1 = repmat({T},1,q);                     
M = spblkdiag(T1{:});                      
lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);         
II = sparse(eye(p));                     
A = repmat(II,1,q);                       
b=ones(p,1);
M = (M+M');
options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
Outputs= quadprog(M, [], [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q);                       
                                                
end
