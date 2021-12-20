% clc;
% clearvars;
% close all;

%% Load Data
A=[x1,x2,x3,x4,x5,y1]';
index = randsample(1:length(A), 50);
x = A(1:5,index);
t = A(6,index);

data=[x', t'];

TrainInputs=x';
TrainTargets=t';
TrainData=[TrainInputs TrainTargets];


xx= A(1:5,:);
tt= A(6,:);

TestInputs=xx';
TestTargets=tt';
TestData=[TestInputs TestTargets];


%% Design ANFIS


        Prompt={'Influence Radius:'};
        Title='Enter genfis2 parameters';
        DefaultValues={'0.6'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        
        Radius=str2num(PARAMS{1}); %#ok
        
        fis=genfis2(TrainInputs,TrainTargets,Radius);
        


Prompt={'Maximum Number of Epochs:',...
        'Error Goal:',...
        'Initial Step Size:',...
        'Step Size Decrease Rate:',...
        'Step Size Increase Rate:'};
Title='Enter genfis3 parameters';
DefaultValues={'300','0','0.01','0.9','1.1'};

PARAMS=inputdlg(Prompt,Title,1,DefaultValues);


MaxEpoch=str2num(PARAMS{1});                %#ok
ErrorGoal=str2num(PARAMS{2});               %#ok
InitialStepSize=str2num(PARAMS{3});         %#ok
StepSizeDecreaseRate=str2num(PARAMS{4});    %#ok
StepSizeIncreaseRate=str2num(PARAMS{5});    %#ok
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis(TrainData,fis,TrainOptions,DisplayOptions,[],OptimizationMethod);

%% Apply ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);


%% Apply ANFIS to Test Data

TestOutputs=evalfis(TestInputs,fis);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);


%% graph

PlotResults1(TestTargets, TestOutputs,'BTS');
%% out
% out=[TestTargets,TestOutputs];
% xlswrite('anfis.xls',out);

%% PLOT MF
figure;
[x,mf] = plotmf(fis,'input',1);
subplot(5,1,1)
plot(x,mf)
xlabel('Membership Functions for "Vp"')
[x,mf] = plotmf(fis,'input',2);
subplot(5,1,2)
plot(x,mf)
xlabel('Membership Functions for "D"')
[x,mf] = plotmf(fis,'input',3);
subplot(5,1,3)
plot(x,mf)
xlabel('Membership Functions for "A"')
ylabel('Degree of membership', 'FontSize',12,'fontweight','bold');
[x,mf] = plotmf(fis,'input',4);
subplot(5,1,4)
plot(x,mf)
xlabel('Membership Functions for "P"')
[x,mf] = plotmf(fis,'input',5);
subplot(5,1,5)
plot(x,mf)
xlabel('Membership Functions for "PLI"')
