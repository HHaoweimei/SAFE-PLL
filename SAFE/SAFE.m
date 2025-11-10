%function SAFE
function [ accuracy_test] = SAFE(train_data,train_p_target,test_data,test_target,train_target,k,Maxiter, gamma, lambda, alpha, beta)


kdtree = KDTreeSearcher(normr(train_data));
[XKTrain,XKTest] = Kernelize(train_data, test_data,size(train_data,1));

X=XKTrain;
test_data=XKTest;

[p,q]=size(train_p_target);
[m, d] = size(X);


Y=train_p_target;
X2=zeros(m,d);

W = ones(d,q);
b=ones(q,1);
n1=ones(m, 1);
L=eye(d,d);

P=build_label_manifold(train_data, train_p_target, k,kdtree);
G = eye(p,p);
M=zeros(m,m);
L1=eye(d,d);

A1=X'*X; 

for j = 1:Maxiter
    XA=X+X2;
    W=(A1+X'*X2+X2'*X+X2'*X2+alpha*A1+lambda*L)\(X'*Y+X2'*Y+alpha*X'*P-2*(X'+X2')*n1*b'-alpha*X'*n1*b');

    b=(alpha*P'*n1+2*Y'*n1-W'*(2*XA'+alpha*X')*n1)/((4+alpha)*m);

    XW=X*W;
    H=XW + n1 * b';
    X2 = solve_hat_X(X, W, b, Y, M, gamma,n1,L1);
    G=obtain_G(train_data, P, k,beta,kdtree);
    M = updateM(train_data,train_p_target,k,gamma,beta);
    P=updateP(G,Y,alpha,beta,H);


    test_outputs = test_data* W + repmat(b', size(test_data,1), 1);
    accuracy_test = CalAccuracy(test_outputs, test_target);
    fprintf('The accuracy of SAFE is: %f \n', accuracy_test);

end
end