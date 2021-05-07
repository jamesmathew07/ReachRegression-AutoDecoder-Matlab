% this works only in matlab 2018a,
% in 2019 it shows error, because I made modification in Autoencoder class
clc;close all;clear all;

%%
load('SimuData2.mat');
%     Dat2.In(j,:) = [x(1,:),x(2,:),x(3,:),x(4,:),u(1,:),u(2,:)];
%     Dat2.Out(j,:) = [ initial', target'];
% for i = 1:size(Dat2.In,1)
% xTra{1,i} = Dat2.In(i,:);
% end
xTra = Dat2.In';
tTra = Dat2.Out';

%% visualise data
for i= 1:160
    % plot path in joint space
%     p1 = xTra{1,i}(1:51);
%     p2 = xTra{1,i}(52:102);
    p1 = xTra(1:51,i);
    p2 = xTra(52:102,i);
    p3 = xTra(103:153,i);
    p4 = xTra(154:204,i);
    subplot(131);plot(tTra(3,i),tTra(4,i),'or'); hold on;
    subplot(132);plot(p1,p2,'k');
    hold on;
    plot(tTra(3,i),tTra(4,i),'o'); hold on;
    subplot(133);plot(p3,p4,'k');hold on;
    
    
end
subplot(131);axis image;
xlabel('POS shoulder (rad)');
ylabel('POS elbow (rad)');
subplot(132);axis image;
xlabel('POS shoulder (rad)');
ylabel('POS elbow (rad)');
subplot(133);axis image;
xlabel('Vel shoulder (rad/s)');
ylabel('Vel elbow (rad/s)');

%% Training the first autoencoder
hiddenSize1 = 100;
[autoenc1 autonet1 net1] = trainAutoencoder2(xTra,hiddenSize1,'MaxEpochs',3000);
%     , ...
%     'L2WeightRegularization',0.004, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.15, ...
%     'ScaleData', false);
view(autoenc1)
decoder1 = getDe(autoenc1,net1);
view(decoder1)
% Visualizing the weights of the first autoencoder
% figure()
% plotWeights(autoenc1);
feat1 = encode(autoenc1,xTra);

%% Training the second autoencoder
hiddenSize2 = 50;
[autoenc2 autonet2 net2]  = trainAutoencoder2(feat1,hiddenSize2 );%, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
view(autoenc2)
decoder2 = getDe(autoenc2,net2);
view(decoder2)
feat2 = encode(autoenc2,feat1);

%% Training the final softmax layer
% 
% softnet = trainSoftmaxLayer(feat2,tTra,'MaxEpochs',400);
% view(softnet)
% % Forming a stacked neural network
% stackednet = stack(autoenc1,autoenc2,softnet);
% view(stackednet)

% %% Test
% j = 1;
% for i = 1:8:size(Dat2.In,1)
% xTes(:,j) = Dat2.In(i,:);
% tTes(:,j) = Dat2.Out(i,:);
% j = j+1;
% end
% 
% y = stackednet(xTes);
% plotconfusion(tTes,y);
% 
% %% Fine tuning the stacked neural network
% for i = 1:size(Dat2.In,1)
% xTra2(:,i) = Dat2.In(i,:);
% tTra2(:,i) = Dat2.Out(i,:);
% end
% stackednet = train(stackednet,xTra2,tTra2);
% y = stackednet(xTes);
% plotconfusion(tTes,y);

%
%% Training the final layer
hiddenSize3 = 10;
[autoenc3 autonet3 net3]  = trainAutoencoder2(feat2,hiddenSize3); %, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
view(autoenc3)
decoder3 = getDe(autoenc3,net3);
view(decoder3)
feat3 = encode(autoenc3,feat2);

%% Forming a stacked neural network

% stackednetEN = stack(autoenc1,autoenc2,autoenc3);
% view(stackednetEN)
% 

%%
% stackednetDE = stack(decoder3,decoder2,decoder1);
% view(stackednetDE)


% y = stackednetDE(xTra);
% plotconfusion(tTra,y);

% net   = network(stackednet);
% Xpred = net(xTra);
% stackednet = train(stackednet,xTra,tTra);

%%
net4 = feedforwardnet(2);
net4 = train(net4,tTra,feat3);
view(net4)
stackednetDE = stack(net4,decoder3,decoder2,decoder1);
view(stackednetDE)
%%
stackednetDE.trainParam.epochs=4000;
deepnet1  = train(stackednetDE, tTra, xTra); 

Out1 = deepnet1(tTra);

netAngToPos = deepnet1;
save netAngToPos

%% visualise  decoded data
for i= 1:160
    % plot path in joint space
    p1 =  Out1(1:51,i);
    p2 =  Out1(52:102,i);
    v1 =  Out1(103:153,i);
    v2 =  Out1(154:204,i);
    subplot(121);plot(p1,p2,'k'); title('Pos');hold on
    subplot(122);plot(v1,v2,'k'); title('Vel');
    hold on;
%     plot(tTra(3,i),tTra(4,i),'o'); hold on;
end
% axis image;
subplot(121);xlabel('shoulder (rad)');
ylabel('elbow (rad)');

%%
        