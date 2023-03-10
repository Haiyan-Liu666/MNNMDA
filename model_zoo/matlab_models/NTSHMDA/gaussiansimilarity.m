function [kd,km] = gaussiansimilarity(interaction,nd,nm)

%calculate gamad for Gaussian kernel calculation
 gamad = nd/(norm(interaction,'fro')^2);

%calculate Gaussian kernel for the similarity between diseases: kd
C=interaction;
kd=zeros(nd,nd);
D=C*C';
for i=1:nd
    for j=i:nd
        kd(i,j)=exp(-gamad*(D(i,i)+D(j,j)-2*D(i,j)));
    end
end
kd=kd+kd'-diag(diag(kd));

%calculate gamam for Gaussian kernel calculation
 gamam = nm/(norm(interaction,'fro')^2);

%calculate Gaussian kernel for the similarity between microbes: km
E=C'*C;
for i=1:nm
    for j=i:nm
        km(i,j)=exp(-gamam*(E(i,i)+E(j,j)-2*E(i,j)));
    end
end
km=km+km'-diag(diag(km));
end