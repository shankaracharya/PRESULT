function ME_train_abs_risk(varargin)
fprintf('\n Example command line: "./presult train_abs_risk ME --train_data <train_data.txt> --randm <yes/no> --id_age <id_age_label.txt>" \n');
p = inputParser;
addParameter(p,'train_data','train_data',@ischar);
addParameter(p,'id_age','id_age.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'out','ME_risk',@ischar);
addParameter(p,'emiters','10');
addParameter(p,'actfunc','logistic',@ischar);
addParameter(p,'hexp','8');
addParameter(p,'hgate','20');
addParameter(p,'nexp','2');
addParameter(p,'mstep_iters','7');
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

train_file = (p.Results.train_data);
var = (p.Results.var);
id_age = (p.Results.id_age);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
n = (p.Results.emiters);
actfunc = (p.Results.actfunc);
out = (p.Results.out);
hexp = (p.Results.hexp);
hgate = (p.Results.hgate);
nexp = (p.Results.nexp);
mstep_iters = (p.Results.mstep_iters);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
tit = (p.Results.title);
n = str2double(n);
hexp = str2double(hexp);
hgate = str2double(hgate);
nexp = str2double(nexp);
mstep_iters = str2double(mstep_iters);

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
net_p =  mixmlp(ninput-2,hexp,1,hgate,nexp,actfunc,'standard');
options  = zeros(1,18);
options(1)  =  -1;  % no display of the error value within the M-step 
options(14) =  mstep_iters;  % the number of iterations in the M-step   
[net,var] = mixmlpem( net_p, x, t, options, n,'scg');
save([out,'_trained_net'],'net');
fprintf(' The ME model saved as: ');
fprintf(out);fprintf('_trained_net.mat \n')
y_train = mixmlpfwd(net,x);
% save([out,'_trainning_score.txt'],'y_train','-ascii')
%%%%%%%%% LOGISTIC REGRESSION Train%%%%%%%%%%%%
[b_train,dev,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train(2:end);% disp(t);
stats.p = stats.p(2:end);
ind_pv = stats.p<=0.01;
x_train = x(:,ind_pv);
b_train_pv = b_train(ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train);
s2_train = sum(ax3_train,2);
save([out,'_log_train_out.txt'],'b_train','-ascii')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_odd_ratio = log(y_train./(1-y_train));
[b_one,dev,stats] = glmfit(log_odd_ratio,t,'binomial','link','logit');
b_one = b_one(2);
fprintf('\n beta one value: %f\r', b_one);
%%%%%%%%%%%%%%%%% Calculation of r_second_term %%%%%%%%%%
ind_affected = find(t==1);
three_percentile = prctile(x,[25 50 75]);
bb_50th_per = three_percentile(2,2:end);
fprintf('\n The median baseline risk values without age saved as: ');
fprintf(out);fprintf('_median_baseline.txt ');
fprintf('\n');
save([out,'_median_baseline.txt'],'bb_50th_per','-ascii');
fprintf('\n median_baseline risk values:');
disp(bb_50th_per);
%%%%%%%%%%%%%%%%% Calculation of r_first_term %%%%%%%%%%
id_age_label_data = load(id_age);
ind_id_age_affected = id_age_label_data(:,3)==1;
all_affected_Age = id_age_label_data(ind_id_age_affected,:);
affected_Age = all_affected_Age(:,2);
affected_y_train = y_train(ind_affected);
size_affected_y_train = size(affected_y_train,1);
fprintf(' number of cases in train dataset: %d\r', size_affected_y_train);
affected_log_odd_ratio = log((affected_y_train)./(1-affected_y_train));
rj_first_term = b_one.*affected_log_odd_ratio;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%Calculation of r_second_term for 50th percentile %%%%%%%%%%%%%%

r_j_cases_matrix_50th_per = [affected_Age(:,1), repmat(bb_50th_per,size(affected_Age,1),1)];
test_case_matrix_50th_per = r_j_cases_matrix_50th_per(:,1:end);
% save([out,'_test_case_matrix_50th_per.txt'],'test_case_matrix_50th_per','-ascii');
y_50th_perc = mixmlpfwd(net, test_case_matrix_50th_per);
affected_log_odd_ratio_50th_perc = (log((y_50th_perc)./(1-y_50th_perc)));
rj_second_term_50th_perc = b_one.*(affected_log_odd_ratio_50th_perc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%calculation of rj and Attributable Risk %%%%%%%%%%%%%%%%%%
rj_50th_per = exp(rj_first_term)./exp(rj_second_term_50th_perc);
sum_of_one_by_rj_50th = sum(1./rj_50th_per);
AR_50th = 1-((1/size_affected_y_train).*sum_of_one_by_rj_50th);
fprintf('\n Attributable risk value: %f\r', AR_50th);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x_ME_train,y_ME_train,T_ME_train,AUC_ME_train] = perfcurve(t,y_train',size(t,2));
[x_logistic_train,y_logistic_train,T_logistic_train,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
if strcmp(log_reg,'on');
    fprintf('\n AUC of the Train Model for ME = %f\r', AUC_ME_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_ME_train, y_ME_train, x_logistic_train, y_logistic_train, '-');
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
    legend_tab_risk = legend(['AUC-ME = ',num2str(AUC_ME_train)], ['AUC-LOG = ',num2str(AUC_Logistic_train)],'Location','SouthEast');
    set(legend_tab_risk,'FontSize',16);
else
    fprintf('\n AUC of the Train Model for ME = %f\r', AUC_ME_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_ME_train, y_ME_train, '-');
    xlabel('1-Specificity');
    ylabel('Sensitivity');
    set(gcf,'color','w');
    grid on;
    box on;
    title(sprintf('AUC=%5.3f', AUC_ME_train));
end
output_train = [out, '_train_ROC'];
export_fig(output_train, '-pdf', '-eps');
end