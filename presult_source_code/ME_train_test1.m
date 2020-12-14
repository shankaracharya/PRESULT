function ME_train_test1(varargin)
fprintf('\n Example command line: "./presult train_test1 ME --input_data <input_data.txt>" \n');
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'emiters','4');
addParameter(p,'actfunc','logistic',@ischar);
addParameter(p,'test_pc','20')
addParameter(p,'var','variables.txt',@ischar)
addParameter(p,'out','ME',@ischar);
addParameter(p,'nexp','5');
addParameter(p,'hexp','8');
addParameter(p,'hgate','20');
addParameter(p,'mstep_iters','7')
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');
addParameter(p,'title','Mixture of Experts',@ischar);

new_array = varargin(:);
for i =1:size(new_array,1)
     for j = 1:size(p.Parameters,2)
         param=p.Parameters{j};
         array_entry=new_array{i};
         first_check=strcat('-',param);
         second_check=strcat('--',param);
         if strcmp(array_entry,first_check)
            new_array{i}=param;
         end
         if strcmp(array_entry,second_check)
            new_array{i}=param;
         end
     end
end
new_array(strcmp('yes',new_array))={'shuffle'};
new_array(strcmp('no',new_array))={'default'};
parse(p,new_array{:});
data_file = (p.Results.input_data);
n = (p.Results.emiters);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
actfunc = (p.Results.actfunc);
var = (p.Results.var);
hexp = (p.Results.hexp);
out = (p.Results.out);
hgate = (p.Results.hgate);
nexp = (p.Results.nexp);
tpc = (p.Results.test_pc);
p_val = (p.Results.pvalue_logistic);
mstep_iters = (p.Results.mstep_iters);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
title_me = (p.Results.title);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_tt1_ME.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_tt1_ME.txt', '--csv','--var',var);
end
data_file1 = load('normalized_data_tt1_ME.txt');
nexp = str2double(nexp);
mstep_iters = str2double(mstep_iters);
n = str2double(n);
hgate = str2double(hgate);
hexp = str2double(hexp);
p_val = str2double(p_val);
tpc = str2double(tpc);
rownumbers = size(data_file1,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DataSorting start                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      %
rng(ran);                                %      %
ran = rand(rownumbers,1);                %      %
data_file2 = [ran data_file1];           %      %
data_file3 = sortrows(data_file2,1);     %      %
data_file = data_file3(:,2:end);         %      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      %
%Data Sorting end                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

chunk = floor(rownumbers*tpc/100);
test = data_file(((floor(100/tpc)-1))*chunk+1:(floor(100/tpc))*chunk,:);
train = [data_file(1:((floor(100/tpc))-1)*chunk,:); data_file((floor(100/tpc))*chunk+1:end, :)];
save([out,'_train_data.txt'], 'train', '-ascii');
save([out,'_test_data.txt'], 'test', '-ascii');
train_input = [out,'_train_data.txt'];
test_input = [out,'_test_data.txt'];
q = load(train_input);
ninput = size(q,2);
x = q(:,1:ninput-1);
t = q(:,ninput);
net =  mixmlp(ninput-1,hexp,1,hgate,nexp,actfunc,'standard');
options  = zeros(1,18);
options(1)  =  -1;  % no display of the error value within the M-step 
options(14) =  mstep_iters;  % the number of iterations in the M-step   
[net,~] = mixmlpem( net, x, t, options, n,'scg');
save ([out,'_trained_net_tt1'],'net');
fprintf('\n Mixture of Experts model file name: ');
ME_model = [out,'_trained_net_tt1.mat'];
disp(ME_model);
test = load(test_input);
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
y_test = mixmlpfwd(net,xtest);
y_train = mixmlpfwd(net,x);
save([out,'_test_score_tt1.txt'], 'y_test', '-ascii');
save([out,'_train_score_tt1.txt'], 'y_train', '-ascii');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[C]=confmat(y_test,ttest);
fprintf('\n Confusion Matrix of the Test Model: \r\n');
disp(C);
C1 = C(1,1);
C2 = C(1,2);
C3 = C(2,1);
C4 = C(2,2);
Correct_prediction = C1+C4;
Wrong_prediction = C2+C3;
Accuracy = (Correct_prediction/(Correct_prediction + Wrong_prediction))*100;
Sensitivity = C1/(C1+C3);
Specificity = C4/(C4+C2);

[x_train,y_train,~,AUC_ME_train] = perfcurve(t, y_train',size(t,2));
[x_test,y_test,~,AUC_ME_test] = perfcurve(ttest,y_test',size(ttest,2));

fprintf(' Sensitivity of the Test Model = %4.3f\r', Sensitivity);
fprintf('\n Specificity of the Test Model = %4.3f\r', Specificity);
fprintf('\n Accuracy of the Test Model = %4.3f\r', Accuracy);
    

%%%%%%%%% LOGISTIC REGRESSION TRAIN%%%%%%%%%%%%

[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
p_value = stats.p;
stats.p = stats.p(2:end);
ind_pv = stats.p<=p_val;
x_train_pv = x(:,ind_pv);
bb_train = b_train(ind_pv);
ax3_train = bsxfun(@times,bb_train',x_train_pv);
s2_train = sum(ax3_train,2);


%%%%%%%% Random Forest %%%%%%%%%%
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
xxtest = xtest(:,ind_pv);

ax3_test = bsxfun(@times,bb_train',xxtest);
s2_test = sum(ax3_test,2);
[x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
fprintf('\n AUC of the RF Train Model = %4.3f\r', AUC_ME_train);        
fprintf('\n AUC of the RF Test Model = %4.3f\r', AUC_ME_test);


%%%%%%%%%%%%%%%

if strcmp(log_reg,'on')
   fprintf('\n AUC of the Logistic Train Model = %4.3f\r', AUC_Logistic_train);
   fprintf('\n AUC of the Logistic Test Model = %4.3f\r', AUC_Logistic_test);
   plot(x_train,y_train,'b-',x_test,y_test,'m-');
   hold all;
   plot(x_logistic_train,y_logistic_train,'r-',x_logistic_test,y_logistic_test,'g-');
   hold off;
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   box on;
   title_tt1 = title(sprintf(title_me));
   set(title_tt1,'FontSize',16)
   xlabel_tt1 = xlabel('1-Specificity');
   set(xlabel_tt1,'FontSize',16);
   ylabel_tt1 = ylabel('Sensitivity');
   set(ylabel_tt1,'FontSize',16);   
   grid on;
   legend_ME_tt1 = legend(sprintf('train-ME =%4.3f',AUC_ME_train), sprintf('test-ME =%4.3f',AUC_ME_test), sprintf('train-log = %4.3f',AUC_Logistic_train), sprintf('test-log =%4.3f',AUC_Logistic_test) ,'Location','SouthEast');
   set(legend_ME_tt1,'FontSize',16);
   save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
   save([out,'_logistic_p_value.txt'],'p_value','-ascii');
   fprintf('\n logistic regression model file name: ');
   logistic_regression_model = [out,'_logistic_regression_model.txt'];
   disp(logistic_regression_model);
   fprintf('\n logistic regression model p_value file name: ');
   logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
   disp(logistic_regression_pvalue);
else
   plot(x_train,y_train,'b-',x_test,y_test,'m-');
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   box on;
   xlabel_tt1 = xlabel('1-Specificity');
   set(xlabel_tt1,'FontSize',16);
   ylabel_tt1 = ylabel('Sensitivity');
   set(ylabel_tt1,'FontSize',16);   
   grid on;
   title_tt1 = title(sprintf(title_me));
   set(title_tt1,'FontSize',16)
   legend_ME_tt1 = legend(sprintf('train-ME =%4.3f',AUC_ME_train), sprintf('test-ME =%4.3f',AUC_ME_test), 'Location','SouthEast');
   set(legend_ME_tt1,'FontSize',16)
end
output_tt1 = [out,'_train_test1'];
export_fig(output_tt1, '-pdf','-eps');
end