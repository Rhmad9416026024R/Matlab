clc;
clearvars;
close all;

%% Load Data
A=[x1,x2,x3,x4,x5,y1];

AllInputs = A(:,1:5);
AllTargets = A(:,6);

indices=(randperm(66))';
Trainingset=A(indices(1:50),:);
Testingset=A(indices(50:end),:);

TrainInputs = Trainingset(:,1:5);
TrainTargets = Trainingset(:,6);

TrainData=[TrainInputs TrainTargets];


TestInputs = Testingset(:,1:5);
TestTargets = Testingset(:,6);

TestData=[TestInputs TestTargets];


x = TrainInputs';
t = TrainTargets';
n=numel(t);

%% Design SVR

epsilon=0.012;

C=19;

gamma=1.3;

Kernel=@(xi,xj) exp(-gamma*norm(xi-xj)^2);

H=zeros(n,n);
for i=1:n
    for j=i:n
        H(i,j)=Kernel(x(:,i),x(:,j));
        H(j,i)=H(i,j);
    end
end

HH=[ H -H
    -H  H];

f=[-t'; t']+epsilon;

Aeq=[ones(1,n) -ones(1,n)];
beq=0;

lb=zeros(2*n,1);
ub=C*ones(2*n,1);

options=optimset('Display','iter','MaxIter',1000);

alpha=quadprog(HH,f,[],[],Aeq,beq,lb,ub,[],options);

alpha=alpha';

AlmostZero=(abs(alpha)<max(abs(alpha))*1e-4);

alpha(AlmostZero)=0;

alpha_plus=alpha(1:n);
alpha_minus=alpha(n+1:end);

eta=alpha_plus-alpha_minus;

S=find(alpha_plus+alpha_minus>0 & alpha_plus+alpha_minus<C);




%% Apply SVR to Train Data

TrainOutputs=zeros(size(t));
for i=1:n
    TrainOutputs(i)=MySVRFunc(x(:,i),eta(S),x(:,S),Kernel);
end

b=mean(t(S)-TrainOutputs(S)-sign(eta(S))*epsilon);

TrainOutputs=TrainOutputs+b;

TrainOutputs=TrainOutputs';

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);


%% Apply SVR to Test Data
TestInputs= TestInputs';
TestTargets= TestTargets';
TestOutputs=zeros(size(TestTargets));
for k=1:numel(TestOutputs)
    TestOutputs(k)=MySVRFunc(TestInputs(:,k),eta(S),x(:,S),Kernel)+b;
end

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

%% Apply SVR to All Data
AllInputs= AllInputs';
AllTargets= AllTargets';
AllOutputs=zeros(size(AllTargets));
for l=1:numel(AllOutputs)
    AllOutputs(l)=MySVRFunc(AllInputs(:,l),eta(S),x(:,S),Kernel)+b;
end

AllTargets= AllTargets';
AllOutputs= AllOutputs';
AllErrors=AllTargets-AllOutputs;
AllMSE=mean(AllErrors(:).^2);
AllRMSE=sqrt(AllMSE);
AllErrorMean=mean(AllErrors);
AllErrorSTD=std(AllErrors);


%% graph

PlotResults1(AllTargets,AllOutputs,'BTS');
PlotResults(TestTargets, TestOutputs,'Test Data');
PlotResults(TrainTargets,TrainOutputs,'Train Data');
PlotResults(AllTargets,AllOutputs,'All Data');

%% out
% out=[AllTargets,AllOutputs];
% xlswrite('svr.xls',out);

