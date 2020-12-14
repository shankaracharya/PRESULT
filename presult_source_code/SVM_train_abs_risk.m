function SVM_train_abs_risk(varargin)
fprintf('\n Example command line: "./presult train_abs_risk SVM --train_data <train_data.txt> --randm <yes/no> --id_age <id_age_label.txt>" \n');
p = inputParser;
addParameter(p,'train_data','pid_raw_data_wd_id.txt',@ischar);
addParameter(p,'id_age','pid_id_age_label.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'out','SVM_risk',@ischar);
addParameter(p,'kernel_function','linear',@ischar); % 'rbf' or 'linear'
addParameter(p,'iters_limit','300'); % optimize iterations limit for input data
addParameter(p,'box_constraint','1'); % BoxConstraint 1 for classification 
addParameter(p,'CacheSize','maximum',@ischar);
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');
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

train_file = (p.Results.train_data);
var = (p.Results.var);
id_age = (p.Results.id_age);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
out = (p.Results.out);
box_constraint = (p.Results.box_constraint);
kernel_function = (p.Results.kernel_function);
iters_limit = (p.Results.iters_limit);
ran = (p.Results.randm);
tit = (p.Results.title);
log_reg = (p.Results.logistic_regression);
box_constraint = str2double(box_constraint);
iters_limit = str2double(iters_limit);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_ME.txt','-var',var,'--with_id');

else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_ME.txt', '--csv','-var',var,'--with_id');
end


q = load('normalized_train_data_ME.txt');

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
% disp(data_file);
ninput = size(data_file,2);
x = data_file(:,2:ninput-1);
t = data_file(:,ninput);
SVMModel = fitcsvm(x,t,'KernelFunction',kernel_function,'Standardize',true, 'IterationLimit',iters_limit,'BoxConstraint',box_constraint,'Verbose',0);
[ScoreSVMModel] = fitSVMPosterior(SVMModel,x,t);
save([out,'_trained_net'],'SVMModel');
[~,score_train] = predict(ScoreSVMModel,x);
fprintf('\n The SVM model saved as: ');
fprintf(out);fprintf('_trained_net.mat ')
y_train = score_train(:,2);
% disp(y_train);
save([out,'_trainning_score.txt'],'y_train','-ascii')
%%%%%%%%% LOGISTIC REGRESSION Train%%%%%%%%%%%%
[b_train,dev,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train(2:end);% disp(t);
stats.p = stats.p(2:end);
ind_pv = stats.p<=0.05;
x_train = x(:,ind_pv);
b_train_pv = b_train(ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train);
s2_train = sum(ax3_train,2);
save([out,'_log_train_out.txt'],'b_train','-ascii')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_odd_ratio = log(y_train./(1-y_train));
[b_one,dev,stats] = glmfit(log_odd_ratio,t,'binomial','link','logit');
b_one = b_one(2);
fprintf('\n');
fprintf('\n beta one value: %f\r', b_one);
%%%%%%%%%%%%%%%%% Calculation of r_second_term %%%%%%%%%%
ind_affected = t==1;
three_percentile = prctile(x,[25 50 75]);
bb_50th_per = three_percentile(2,2:end);
fprintf('\n The median baseline risk values without age saved as: ');
fprintf(out);fprintf('_median_baseline.txt ');
fprintf('\n');
save([out,'_median_baseline.txt'],'bb_50th_per','-ascii');
fprintf('\n median_baseline risk values without age:');
disp(bb_50th_per);
%%%%%%%%%%%%%%%%% Calculation of r_first_term %%%%%%%%%%
id_age_label_data = load(id_age);
ind_id_age_affected = id_age_label_data(:,3)==1;
all_affected_Age = id_age_label_data(ind_id_age_affected,:);
affected_Age = all_affected_Age(:,1);
affected_y_train = y_train(ind_affected);
size_affected_y_train = size(affected_y_train,1);
fprintf(' number of cases in train dataset: %d\r', size_affected_y_train);
affected_log_odd_ratio = log((affected_y_train)./(1-affected_y_train));
rj_first_term = b_one.*affected_log_odd_ratio;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%Calculation of r_second_term for 50th percentile %%%%%%%%%%%%%%
r_j_cases_matrix_50th_per = [affected_Age(:,1), repmat(bb_50th_per,size(affected_Age,1),1)];
test_case_matrix_50th_per = r_j_cases_matrix_50th_per(:,1:end);
% save([out,'_test_case_matrix_50th_per.txt'],'test_case_matrix_50th_per','-ascii')
[ScoreSVMModel_y,~] = fitPosterior(SVMModel);
[~,score_50th_per] = predict(ScoreSVMModel_y,test_case_matrix_50th_per);
y_50th_per = score_50th_per(:,1);
affected_log_odd_ratio_50th_perc = (log((y_50th_per)./(1-y_50th_per)));
rj_second_term_50th_perc = b_one.*(affected_log_odd_ratio_50th_perc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%calculation of rj and Attributable Risk %%%%%%%%%%%%%%%%%%
rj_50th_per = exp(rj_first_term)./exp(rj_second_term_50th_perc);
sum_of_one_by_rj_50th = sum(1./rj_50th_per);
AR_50th = 1-((1/size_affected_y_train).*sum_of_one_by_rj_50th);
fprintf('\n Attributable risk value: %f\r', AR_50th);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x_SVM_train,y_SVM_train,T_SVM_train,AUC_SVM_train] = perfcurve(t,y_train',size(t,2));
[x_logistic_train,y_logistic_train,T_logistic_train,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
if strcmp(log_reg,'on');
    fprintf('\n AUC of the Train Model for SVM = %f\r', AUC_SVM_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_SVM_train, y_SVM_train, x_logistic_train, y_logistic_train, '-');
    hold off;
    xlabel_tt1 = xlabel('1-Specificity');
    set(xlabel_tt1,'FontSize',18);
    ylabel_tt1 = ylabel('Sensitivity');
    set(ylabel_tt1,'FontSize',18);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    box on;
    title_tt1 = title(sprintf(tit));
    set(title_tt1,'FontSize',16);
    legend_tab_risk = legend(['AUC-SVM = ',num2str(AUC_SVM_train)], ['AUC-LOG = ',num2str(AUC_Logistic_train)],'Location','SouthEast');
    set(legend_tab_risk,'FontSize',16);
else
    fprintf('\n AUC of the Train Model for SVM = %f\r', AUC_SVM_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_SVM_train, y_SVM_train, '-');
    xlabel_tt1 = xlabel('1-Specificity');
    set(xlabel_tt1,'FontSize',16);
    ylabel_tt1 = ylabel('Sensitivity');
    set(ylabel_tt1,'FontSize',16);
    set(gcf,'color','w');
    grid on;
    box on;
    tit(sprintf('AUC=%5.3f', AUC_SVM_train));
end
output_train = [out, '_train_ROC'];
export_fig(output_train, '-pdf', '-eps');
end