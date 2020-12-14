function SVM_yearly_risk(varargin)
fprintf('\n Example command line: "./presult yearly_risk SVM --trained_net <SVM_risk_trained_net.mat> --baseline_risk <baseline_risk.txt> --AR <float> --predictive_data <prediction_data.txt> --id_age_data <id_age_data.txt> --beta_one <float> --incidance_rate <inc_rate_blocks.txt>" \n');
p = inputParser;
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'id_age','pid_id_age_label.txt',@ischar);
addParameter(p,'trained_net','SVM_trained_net.mat');
addParameter(p,'baseline_risk','baseline_risk.txt',@ischar);
addParameter(p,'out','SVM',@ischar);
addParameter(p,'AR','0.5');
addParameter(p,'beta_one','1');
addParameter(p,'risk_years','10');
addParameter(p,'predictive_data','pid_prediction_data.txt',@ischar);
addParameter(p,'incidence_rate','inc_mort_rate.txt',@ischar);
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
csv = (p.Results.csv);
miss = (p.Results.missing_value);
b_one = (p.Results.beta_one);
SVMModel = (p.Results.trained_net);
AR = (p.Results.AR);
id_age = (p.Results.id_age);
base_risk=(p.Results.baseline_risk);
out = (p.Results.out);
risk_years = (p.Results.risk_years);
predict_data = (p.Results.predictive_data);
inc_rate = (p.Results.incidence_rate);
AR = str2double(AR);
b_one = str2double(b_one);
risk_years = str2double(risk_years);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', predict_data, '--output', 'predict_data_normalized.txt','--with_id');
else
   perl('newest_normalization.pl', '--input', predict_data, '--output', 'predict_data_normalized.txt', '--csv','--with_id');
end
%%%%%%%%%%%%%%%% Calculation of r_first_term %%%%%%%%%

predictive_test = load('predict_data_normalized.txt');
xtest_pred = predictive_test(:,2:end-1);
load(SVMModel,'-mat');
ScoreSVMModel = fitSVMPosterior(SVMModel);
[~,score_test] = predict(ScoreSVMModel,xtest_pred);
y_test_pred = score_test(:,2);
predictive_ratio = (y_test_pred)./(1-y_test_pred);
predictive_log_ratio = log(predictive_ratio);
r_first_term = b_one.*predictive_log_ratio;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%Calculating r_second_term %%%%%%%%%%%%%%%%%%%%
base_risk=load(base_risk);
base_risk = [20, base_risk];
prediction_matrix = repmat(base_risk,size(xtest_pred,1),1);
[~,score_pred_mat] = predict(ScoreSVMModel,prediction_matrix);
y_pred_mat = score_pred_mat(:,2);
r_second_term = b_one.*(log((y_pred_mat)./(1-y_pred_mat)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Calculating ri %%%%%%%%%%%%%%%%%%%%
ri = exp(r_first_term - r_second_term);
save([out,'_ri.txt'], 'ri', '-ascii');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
id_and_ri = [predictive_test(:,1) ri];
save([out,'_id_and_ri.txt'], 'id_and_ri', '-ascii');
inc = load(inc_rate);
bs = inc(2,1) - inc(1,1);
if bs > 1
   lowest_age = inc(1,1);
   highest_age = inc(end,1);
   block_count = (highest_age - lowest_age)/bs;
   inc1 = (inc(1,1):inc(end,1)+ (bs-1))';
   for i = 1:(block_count+1)
      inc2(i,:) = [repmat(inc(i,2),bs,1)];
   end
   inc22 = inc2';
   inc222 = inc22(:);
   for i = 1:(block_count+1)
       inc3(i,:) = [repmat(inc(i,3),bs,1)];
   end
   inc33 = inc3';
   inc333 = inc33(:);
   incidance_rate_table_male = [inc1 inc222 inc333];
else
   incidance_rate_table_male = inc;
end
incidance_rate_table_male_age = incidance_rate_table_male(:,1);
max_incid_age = max(incidance_rate_table_male_age);
min_incid_age = min(incidance_rate_table_male_age);
index_min_age= find(incidance_rate_table_male_age==min_incid_age);
index_max_age= find(incidance_rate_table_male_age==max_incid_age);    
incidence_irt_male_pre = incidance_rate_table_male(:,2);
incidence_irt_male = incidence_irt_male_pre;
mortality_irt_male = incidance_rate_table_male(:,3);
if min_incid_age >1
    age_append = 1:(min_incid_age-1);
    incid_append = repmat(incidence_irt_male(index_min_age), (min_incid_age-1), 1);
    mort_append = repmat(mortality_irt_male(index_min_age), (min_incid_age-1), 1);
end

append_all_min = [age_append' incid_append mort_append];
if max_incid_age <=160
   max_age_append = ((max_incid_age+1):160);
   max_incid_append = repmat(incidence_irt_male(index_max_age), (160-max_incid_age), 1); 
   max_mort_append = repmat(mortality_irt_male(index_max_age), (160-max_incid_age), 1);
end


append_all_max = [max_age_append' max_incid_append max_mort_append];
rate_matrix_including_below_min = [append_all_min; incidance_rate_table_male];
final_inc_table = [rate_matrix_including_below_min; append_all_max];
id_age = load(id_age);
[lib,locb] = ismember(id_and_ri(:,1),id_age(:,1));
only_age = id_age(:,2);
retrieved_age = only_age(locb);
r = ri;
for i = 2:risk_years
    retrieved_age(:,i) = retrieved_age(:,i-1) + 1 ;
end

[libbb,locbbb] = ismember(retrieved_age,final_inc_table(:,1));

mmm = final_inc_table(:,2);
incidence_select_age2 = mmm(locbbb);

nnn = final_inc_table(:,3);
mortality_select_age2 = nnn(locbbb);
h11 = incidence_select_age2.*(1-AR);
save([out,'_h11.txt'], 'h11', '-ascii');
h22 = mortality_select_age2;

for i=1:risk_years
    h11_times_r(:,i) = h11(:,i).*r;
end
save([out,'_h11_times_r.txt'], 'h11_times_r', '-ascii');
p_all = (h11_times_r./(h11_times_r + h22).*(1-(exp(-(h11_times_r + h22)))));
no_of_rows_zeros = size(p_all,1);
no_of_columns_zeros = size(p_all,2) - 1;
p_final = [p_all(:,1), zeros(no_of_rows_zeros,no_of_columns_zeros)];

for i = 2:size(h11_times_r,2)
     p_final(:,i) = (h11_times_r(:,i)./(h11_times_r(:,i) + h22(:,i))).*(1-(exp(-(h11_times_r(:,i) + h22(:,i))))).*((exp(-(r.*(sum((h11(:,1:i-1)),2)))-(sum((h22(:,1:i-1)),2)))));
     p_final(:,i) = plus(p_final(:,i-1),p_final(:,i)); 
end
disp(p_final);
save([out,'_yearwise_absolute_risk.txt'], 'p_final', '-ascii');
end





