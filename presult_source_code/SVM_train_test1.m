function SVM_train_test1(varargin)
fprintf('\n Example command line: "./presult train_test1 SVM --input_data <input_data.txt>" \n');
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'box_constraint','1'); % BoxConstraint 1 for classification 
addParameter(p,'CacheSize','maximum',@ischar);
addParameter(p,'kernel_function','linear',@ischar); % 'rbf' or 'linear'
addParameter(p,'out','SVM',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'test_pc','20');
addParameter(p,'iters_limit','300'); % optimize iterations limit for input data
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'title','Support Vector Machine',@ischar);

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
box_constraint = (p.Results.box_constraint);
kernel_function = (p.Results.kernel_function);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
out = (p.Results.out);
tpc = (p.Results.test_pc);
iters_limit = (p.Results.iters_limit);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
title_svm = (p.Results.title);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_tt1_SVM.txt');
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_tt1_SVM.txt', '--csv');
end

data_file1 = load('normalized_data_tt1_SVM.txt');

box_constraint = str2double(box_constraint);
tpc = str2double(tpc);
iters_limit = str2double(iters_limit);
rownumbers = size(data_file1,1);
%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran data_file1];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
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
SVMModel = fitcsvm(x,t,'KernelFunction',kernel_function,'Standardize',true, 'IterationLimit',iters_limit,'BoxConstraint',box_constraint,'Verbose',0);
test = load(test_input);
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
ScoreSVMModel = fitPosterior(SVMModel,x,t);
[~,score_test] = predict(ScoreSVMModel,xtest);
save ([out,'_trained_net_tt1'],'ScoreSVMModel');
y_svm_test = score_test(:,2);
[~,score_train] = predict(ScoreSVMModel,x);
y_svm_train = score_train(:,2);
save([out,'_test_score_tt1.txt'], 'y_svm_test', '-ascii');
save([out,'_train_score_tt1.txt'], 'y_svm_train', '-ascii');
[C_svm]=confmat(y_svm_test,ttest);
fprintf('\n Confusion Matrix of the Test Model: \r\n');
disp(C_svm);
C1 = C_svm(1,1);
C2 = C_svm(1,2);
C3 = C_svm(2,1);
C4 = C_svm(2,2);
Correct_prediction_SVM = C1+C4;
Wrong_prediction_SVM = C2+C3;
Accuracy = (Correct_prediction_SVM/(Correct_prediction_SVM + Wrong_prediction_SVM))*100;
Sensitivity = C1/(C1+C3);
Specificity = C4/(C4+C2);
fprintf(' Sensitivity of the SVM Test Model = %4.3f\r', Sensitivity);
fprintf('\n Specificity of the SVM Test Model = %4.3f\r', Specificity);
fprintf('\n Accuracy of the SVM Test Model = %4.3f\r', Accuracy);
%%%%%%%%%%%%%%%%%%%%% LOGISTIC REGRESSION TRAIN%%%%%%%%%%%%%

[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
p_value = stats.p;
stats.p = stats.p(2:end);
ind_pv = stats.p<=0.01;
x_train_pv = x(:,ind_pv);
bb_train = b_train(ind_pv);
ax3_train = bsxfun(@times,bb_train',x_train_pv);
s2_train = sum(ax3_train,2);


[x_train,y_train,~,AUC_SVM_train] = perfcurve(t, y_svm_train',size(t,2));
[x_test,y_test,~,AUC_SVM_test] = perfcurve(ttest,y_svm_test',size(ttest,2));

[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
xxtest = xtest(:,ind_pv);
ax3_test = bsxfun(@times,bb_train',xxtest);
s2_test = sum(ax3_test,2);
[x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));

fprintf('\n AUC of the SVM Train Model = %4.3f\r', AUC_SVM_train);        
fprintf('\n AUC of the SVM Test Model = %4.3f\r', AUC_SVM_test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(log_reg,'on')
   plot(x_train,y_train,'b-',x_test,y_test,'m-');
   hold all;
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   plot(x_logistic_train,y_logistic_train,'r-',x_logistic_test,y_logistic_test,'g-');
   hold off;
   xlabel_tt1 = xlabel('1-Specificity');
   set(xlabel_tt1,'FontSize',16);
   ylabel_tt1 = ylabel('Sensitivity');
   set(ylabel_tt1,'FontSize',16);
   grid on;
   box on;
   fprintf('\n AUC of the Logistic Train Model = %4.3f\r', AUC_Logistic_train);
   fprintf('\n AUC of the Logistic Test Model = %4.3f\r', AUC_Logistic_test);
   title_tt1 = title(sprintf(title_svm));
   set(title_tt1,'FontSize',16);
   legend_tt2 = legend(sprintf('train-SVM =%4.3f',AUC_SVM_train), sprintf('test-SVM =%4.3f',AUC_SVM_test), sprintf('train-log =%4.3f',AUC_Logistic_train), sprintf('test-log =%4.3f',AUC_Logistic_test),'Location','SouthEast');
   set(legend_tt2,'FontSize',16);
   save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
   fprintf('\n logistic regression model file name: ');
   logistic_regression_model = [out,'_logistic_regression_model.txt'];
   disp(logistic_regression_model);
   save([out,'_logistic_p_value.txt'],'p_value','-ascii');
   fprintf('\n logistic regression model p_value file name: ');
   logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
   disp(logistic_regression_pvalue);
   
else
   plot(x_train,y_train,'b-',x_test,y_test,'m-');
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   xlabel_tt1 = xlabel('1-Specificity');
   set(xlabel_tt1,'FontSize',16);
   ylabel_tt1 = ylabel('Sensitivity');
   set(ylabel_tt1,'FontSize',16);
   grid on;
   box on;
   title_tt1 = title(sprintf(title_svm));
   set(title_tt1,'FontSize',16);
   legend_tt2 = legend(sprintf('train-SVM =%4.3f',AUC_SVM_train), sprintf('test-SVM =%4.3f',AUC_SVM_test),'Location','SouthEast');
   set(legend_tt2,'FontSize',16);
end
graph = [out,'_train_test1'];
export_fig(graph, '-pdf', '-eps');
  
end
