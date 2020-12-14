function SVM_train_mod(varargin)
p = inputParser;
addParameter(p,'train_data','train_data',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'out','out',@ischar);
addParameter(p,'box_constraint','1'); % BoxConstraint 1 for classification 
addParameter(p,'CacheSize','maximum',@ischar);
addParameter(p,'kernel_function','linear',@ischar); % 'rbf' or 'linear'
addParameter(p,'iters_limit','700'); % optimize iterations limit for input data
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');

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

train_file = (p.Results.train_data);
var = (p.Results.var);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
out = (p.Results.out);
ran = (p.Results.randm);
pval_log = (p.Results.pvalue_logistic);
log_reg = (p.Results.logistic_regression);
box_constraint = (p.Results.box_constraint);
kernel_function = (p.Results.kernel_function);
iters_limit = (p.Results.iters_limit);
box_constraint = str2double(box_constraint);
iters_limit = str2double(iters_limit);
pval_log = str2double(pval_log);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data.txt','-var',var,'--with_id');

else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data.txt', '--csv','-var',var,'--with_id');
end


q = load('normalized_train_data.txt');

rownumbers = size(q,1);
%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran q];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
ninput = size(data_file,2);
x = data_file(:,2:ninput-1);
t = data_file(:,ninput);
save([out,'_t.txt'],'t','-ascii');
SVMModel = fitcsvm(x,t,'KernelFunction',kernel_function,'Standardize',true, 'IterationLimit',iters_limit,'BoxConstraint',box_constraint,'Verbose',0);
ScoreSVMModel = fitSVMPosterior(SVMModel,x,t);
save ([out,'_ScoreSVMModel'],'ScoreSVMModel');
[~,score_train] = predict(ScoreSVMModel,x);
y_train = score_train(:,2);
save([out,'_y_train.txt'],'y_train','-ascii')
%%%%%%%%% LOGISTIC REGRESSION Train%%%%%%%%%%%%
[b_train,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train(2:end);% disp(t);
stats.p = stats.p(2:end);
ind_pv = stats.p<=pval_log;
x_train = x(:,ind_pv);
b_train_pv = b_train(ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train);
s2_train = sum(ax3_train,2);
save([out,'_log_train_out.txt'],'b_train','-ascii')
%%%%%%%% Logistic regression %%%%%%%%%%
%%%%%%%%%%%%%%%%% Calculation of r_second_term %%%%%%%%%%

log_odd_ratio = log(y_train./(1-y_train));
[b_one,~,~] = glmfit(log_odd_ratio,t,'binomial','link','logit');
b_one = b_one(2);
fprintf('\n beta one value: %f\r\n', b_one);
ind_una = t==0;
una_y_train = y_train(ind_una);
size_una_y_train = size(una_y_train,1);
fprintf('\n number of controls in train dataset: %d\r\n', size_una_y_train);
b_one_one = b_one/size_una_y_train;
unaffected_log_odd_ratio = log((una_y_train)./(1-una_y_train));
sum_unaffected_log_odd_ratio = sum(unaffected_log_odd_ratio);
r_second_term = b_one_one*sum_unaffected_log_odd_ratio;
fprintf('\n r_second_term values: %f\r\n', r_second_term);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Calculation of r_first_term %%%%%%%%%%
ind_affected = t==1;
affected_y_train = y_train(ind_affected);
size_affected_y_train = size(affected_y_train,1);
fprintf('\n number of cases in train dataset: %d\r\n', size_affected_y_train);
affected_odd_ratio = (affected_y_train)./(1-affected_y_train);
affected_log_odd_ratio = log(affected_odd_ratio);
r_first_term = b_one.*affected_log_odd_ratio;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%calculation of rj and Attributable Risk %%%%%%%%%%%%%%%%%%
rj = exp(r_first_term-r_second_term);
sum_of_one_by_rj = sum(1./rj);
AR = 1-((1/size_affected_y_train)*sum_of_one_by_rj);
fprintf('\n Attributable risk value: %f\r\n', AR);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_SVM_train,y_SVM_train,~,AUC_SVM_train] = perfcurve(t,y_train',size(t,2));
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
if strcmp(log_reg,'on');
    fprintf('\n AUC of the Train Model for SVM = %f\r\n', AUC_SVM_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r\n', AUC_Logistic_train);
    plot(x_SVM_train, y_SVM_train, x_logistic_train, y_logistic_train, '-');
    hold off;
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    set(gcf,'color','w');
    grid on;
    box on;
    title(sprintf('SVM Train and Logistic regression Train ROC Curve'));
    legend_svm_Train = legend(sprintf('AUC-SVM = %4.2f',AUC_SVM_train), sprintf('AUC-LOG = %4.2f',AUC_Logistic_train),'Location','SouthEast');
    set(legend_svm_Train,'FontSize',14);
else
    fprintf('\n AUC of the Train Model for SVM = %f\r\n', AUC_SVM_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r\n', AUC_Logistic_train);
    plot(x_SVM_train, y_SVM_train, '-');
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    set(gcf,'color','w');
    grid on;
    box on;
    title(sprintf('SVM Train and Logistic regression Train ROC Curve'));
    legend_svm_Train = legend(sprintf('AUC-SVM = %4.2f',AUC_SVM_train), sprintf('AUC-LOG = %4.2f',AUC_Logistic_train),'Location','SouthEast');
    set(legend_svm_Train,'FontSize',14);
end
output_train = [out, '_train_ROC'];
export_fig(output_train, '-pdf', '-eps', '-png', '-tiff');
end