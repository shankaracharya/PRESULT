function RT_train_abs_risk(varargin)
fprintf('\n Example command line: "./presult train_abs_risk RT --train_data <train_data.txt> --randm <yes/no> --id_age <id_age_label.txt>" \n');
p = inputParser;
addParameter(p,'train_data','pid_raw_data_wd_id.txt',@ischar);
addParameter(p,'id_age','pid_id_age_label.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'out','RT_risk',@ischar);
addParameter(p,'min_parent','140');
addParameter(p,'min_leaf','120');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');
addParameter(p,'title','Regression Tree',@ischar);

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
p_val_log = (p.Results.pvalue_logistic);
out = (p.Results.out);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
title_rt = (p.Results.title);
min_parent = str2double(min_parent);
min_leaf = str2double(min_leaf);
p_val = str2double(p_val_log);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_RT.txt','-var',var,'--with_id');

else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_RT.txt', '--csv','-var',var,'--with_id');
end


q = load('normalized_train_data_RT.txt');

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
rtree = fitrtree(x,t,'MinParent',min_parent,'MinLeaf', min_leaf);
save([out,'_trained_net'],'rtree');
fprintf('\n The RT model saved as: ');
fprintf(out);fprintf('_trained_net.mat ')
y_train = predict(rtree, x);
% disp(y_train);
save([out,'_trainning_score.txt'],'y_train','-ascii')
%%%%%%%%% LOGISTIC REGRESSION Train%%%%%%%%%%%%
[b_train,dev,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train(2:end);% disp(t);
stats.p = stats.p(2:end);
ind_pv = stats.p<=p_val;
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
fprintf('\n median_baseline risk values without age: ');
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
y_50th_per = predict(rtree,test_case_matrix_50th_per);
affected_log_odd_ratio_50th_perc = (log((y_50th_per)./(1-y_50th_per)));
rj_second_term_50th_perc = b_one.*(affected_log_odd_ratio_50th_perc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%calculation of rj and Attributable Risk %%%%%%%%%%%%%%%%%%
rj_50th_per = exp(rj_first_term)./exp(rj_second_term_50th_perc);
sum_of_one_by_rj_50th = sum(1./rj_50th_per);
AR_50th = 1-((1/size_affected_y_train).*sum_of_one_by_rj_50th);
fprintf('\n Attributable risk value: %f\r', AR_50th);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x_RT_train,y_RT_train,T_SVM_train,AUC_RT_train] = perfcurve(t,y_train',size(t,2));
[x_logistic_train,y_logistic_train,T_logistic_train,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
if strcmp(log_reg,'on');
    fprintf('\n AUC of the Train Model for RT = %f\r', AUC_RT_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_RT_train, y_RT_train, x_logistic_train, y_logistic_train, '-');
    hold off;
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_rt));
    set(title_t,'FontSize',16);
    legend_rt = legend(['AUC-RT = ',num2str(AUC_RT_train)], ['AUC-LOG = ',num2str(AUC_Logistic_train)],'Location','SouthEast');
    set(legend_rt,'FontSize',16);
else
    fprintf('\n AUC of the Train Model for RT = %f\r', AUC_RT_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_RT_train, y_RT_train, '-');
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_rt));
    set(title_t,'FontSize',16);
    legend_rt = legend(['AUC-RT = ',num2str(AUC_RT_train)], 'Location','SouthEast');
    set(legend_rt,'FontSize',16);
end
output_train = [out, '_train_ROC'];
export_fig(output_train, '-pdf', '-eps');
end