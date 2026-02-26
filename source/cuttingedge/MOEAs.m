
clear; clc; close all;
rng(123);

%% 1. 全局参数
global_params = struct();
global_params.pop_size = 80;        % 适度增大种群，配合拥挤度排序
global_params.max_iter = 100;
global_params.n_obj = 2;
global_params.repeat_times = 5;
global_params.per_class_samples = 20;

%% 2. 5类数据集配置
dataset_configs = {
    struct('name','小规模结构化','n_var',10,'sparsity',0.0,'type','structured','func',@calc_ZDT2);
    struct('name','大规模结构化','n_var',50,'sparsity',0.0,'type','structured','func',@calc_ZDT2);
    struct('name','小规模稀疏随机','n_var',15,'sparsity',0.8,'type','random','func',@calc_ZDT3);
    struct('name','大规模稠密随机','n_var',100,'sparsity',0.1,'type','random','func',@calc_ZDT4);
    struct('name','混合性质','n_var',30,'sparsity',0.5,'type','hybrid','func',@calc_Hybrid);
};
n_dataset = length(dataset_configs);

%% 3. 初始化结果矩阵
igd_results = zeros(n_dataset, global_params.per_class_samples, global_params.repeat_times);

%% 4. 遍历数据集运行
for dataset_idx = 1:n_dataset
    disp(['================ 处理第', num2str(dataset_idx), '类：', dataset_configs{dataset_idx}.name, ' ================']);
    curr_config = dataset_configs{dataset_idx};
    
    for sample_idx = 1:global_params.per_class_samples
        [lb, ub, weights] = generate_valid_sample(curr_config, sample_idx);
        
        for repeat_idx = 1:global_params.repeat_times
            % 运行【加入拥挤度排序】的MOEAs
            [pareto_x, pareto_f] = MOEAs_with_diversity(global_params, curr_config, lb, ub, weights);
            true_pareto = generate_true_pareto(curr_config);
            igd = calculate_valid_IGD(pareto_f, true_pareto);
            
            igd_results(dataset_idx, sample_idx, repeat_idx) = igd;
            
            if mod(repeat_idx, 5) == 0
                disp(['  样本', num2str(sample_idx), ' - 重复', num2str(repeat_idx), '：IGD=', num2str(igd, '%.6f')]);
            end
        end
    end
end

%% 5. 统计结果
disp(['================ 所有实验完成，统计汇总 ================']);
stat_results = struct();
for dataset_idx = 1:n_dataset
    curr_name = dataset_configs{dataset_idx}.name;
    curr_igd_flat = reshape(igd_results(dataset_idx, :, :), [], 1);
    
    curr_mean = mean(curr_igd_flat);
    curr_var = var(curr_igd_flat);
    
    stat_results(dataset_idx).name = curr_name;
    stat_results(dataset_idx).mean = curr_mean;
    stat_results(dataset_idx).var = curr_var;
    
    disp(['【', curr_name, '】']);
    disp(['  IGD均值：', num2str(curr_mean, '%.6f'), '  |  IGD方差：', num2str(curr_var, '%.8f')]);
end

%% 6. 可视化
visualize_results(igd_results, dataset_configs, stat_results, global_params);

%% ==============================================
% 子函数1：生成有效样本（无变化）
% ==============================================
function [lb, ub, weights] = generate_valid_sample(config, sample_idx)
    n_var = config.n_var;
    sparsity = config.sparsity;
    
    lb = zeros(1, n_var);
    ub = ones(1, n_var);
    
    rng(sample_idx);
    if strcmp(config.type, 'structured')
        weights = linspace(0.1, 1.0, n_var);
    elseif strcmp(config.type, 'random')
        weights = rand(1, n_var);
        weights(weights < sparsity) = 0;
    else
        half = floor(n_var/2);
        structured_part = linspace(0.1, 1.0, half);
        random_part = rand(1, n_var - half);
        random_part(random_part < sparsity) = 0;
        weights = [structured_part, random_part];
    end
end

%% ==============================================
% 核心修改：MOEAs加入拥挤度排序（保证解的多样性，彻底解决单点）
% ==============================================
function [pareto_x, pareto_f] = MOEAs_with_diversity(params, config, lb, ub, weights)
    n_pop = params.pop_size;
    n_iter = params.max_iter;
    n_var = config.n_var;
    
    % 1. 初始化种群（非负、约束0~1）
    pop_x = rand(n_pop, n_var) .* weights;
    pop_x = max(min(pop_x, ub), lb);
    pop_f = config.func(pop_x, params.n_obj);
    
    % 2. 迭代进化（核心：非支配筛选+拥挤度排序，保证多样性）
    for iter = 1:n_iter
        % 步骤1：实数交叉（保证多样性）
        parent_idx = randi(n_pop, n_pop, 2);
        cross_x = pop_x(parent_idx(:,1),:) .* rand(n_pop,n_var) + pop_x(parent_idx(:,2),:) .* (1-rand(n_pop,n_var));
        cross_x = cross_x .* weights;
        
        % 步骤2：变异+约束
        mutate_x = pop_x + 0.05 * randn(n_pop, n_var); % 恢复合理变异幅度
        mutate_x = mutate_x .* weights;
        mutate_x = max(min(mutate_x, ub), lb);
        
        % 步骤3：合并种群（父代+子代，扩大选择范围）
        new_x = [pop_x; cross_x; mutate_x];
        new_x = max(min(new_x, ub), lb);
        new_f = config.func(new_x, params.n_obj);
        
        % 步骤4：【关键修改1】非支配筛选，得到所有非支配解（帕累托解）
        pareto_idx = is_pareto_optimal(new_f);
        pareto_x_temp = new_x(pareto_idx, :);
        pareto_f_temp = new_f(pareto_idx, :);
        
        % 步骤5：【关键修改2】拥挤度排序（保证解的分散性，避免单点）
        if size(pareto_x_temp, 1) >= n_pop
            % 计算拥挤度，选择拥挤度大的解（更分散）
            crowding_dist = calculate_crowding_distance(pareto_f_temp);
            % 按拥挤度降序排序，取前n_pop个
            [~, sorted_idx] = sort(crowding_dist, 'descend');
            pop_x = pareto_x_temp(sorted_idx(1:n_pop), :);
            pop_f = pareto_f_temp(sorted_idx(1:n_pop), :);
        else
            % 补充随机解，同时计算拥挤度保证分散
            rand_x = rand(n_pop - size(pareto_x_temp, 1), n_var) .* weights;
            rand_x = max(min(rand_x, ub), lb);
            rand_f = config.func(rand_x, params.n_obj);
            
            pop_x = [pareto_x_temp; rand_x];
            pop_f = [pareto_f_temp; rand_f];
            
            % 对补充后的种群计算拥挤度，轻微排序
            crowding_dist = calculate_crowding_distance(pop_f);
            [~, sorted_idx] = sort(crowding_dist, 'descend');
            pop_x = pop_x(sorted_idx, :);
            pop_f = pop_f(sorted_idx, :);
        end
    end
    
    % 3. 提取最终帕累托解集
    final_pareto_idx = is_pareto_optimal(pop_f);
    pareto_x = pop_x(final_pareto_idx, :);
    pareto_f = pop_f(final_pareto_idx, :);
end

%% ==============================================
% 计算拥挤度
% ==============================================
function crowding_dist = calculate_crowding_distance(f)
    [n_samples, n_obj] = size(f);
    crowding_dist = zeros(n_samples, 1);
    
    % 对每个目标函数分别计算
    for obj = 1:n_obj
        % 按当前目标函数排序
        [f_sorted, sorted_idx] = sort(f(:, obj));
        [~, original_idx] = sort(sorted_idx);
        
        % 边界解的拥挤度设为无穷大（优先保留）
        crowding_dist(sorted_idx(1)) = inf;
        crowding_dist(sorted_idx(end)) = inf;
        
        % 计算中间解的拥挤度
        if n_samples > 2
            obj_range = f_sorted(end) - f_sorted(1);
            if obj_range == 0
                obj_range = 1e-6; % 避免除0
            end
            for i = 2:n_samples-1
                crowding_dist(sorted_idx(i)) = crowding_dist(sorted_idx(i)) + (f_sorted(i+1) - f_sorted(i-1)) / obj_range;
            end
        end
    end
end

%% ==============================================
%目标函数（优化数值精度，避免支配判断失效）
% ==============================================
function f = calc_ZDT2(x, n_obj)
    n_samples = size(x, 1);
    f = zeros(n_samples, n_obj);
    for i = 1:n_samples
        curr_x = x(i, :);
        f1 = curr_x(1);
        g = 1 + 9 * mean(curr_x(2:end));
        f2 = g * (1 - (f1/g).^2);
        f(i, :) = [f1, f2];
    end
    % 优化归一化：保留微小差异，避免支配判断失效
    f = (f - min(f)) ./ (max(f) - min(f) + 1e-8);
end

function f = calc_ZDT3(x, n_obj)
    n_samples = size(x, 1);
    f = zeros(n_samples, n_obj);
    for i = 1:n_samples
        curr_x = x(i, :);
        f1 = curr_x(1);
        g = 1 + 9 * mean(curr_x(2:end));
        f2 = g * (1 - sqrt(f1/g) - (f1/g) * sin(10*pi*f1));
        f(i, :) = [f1, f2];
    end
    f = (f - min(f)) ./ (max(f) - min(f) + 1e-8);
end

function f = calc_ZDT4(x, n_obj)
    n_samples = size(x, 1);
    f = zeros(n_samples, n_obj);
    for i = 1:n_samples
        curr_x = x(i, :);
        f1 = curr_x(1);
        g = 1 + 10*(size(x,2)-1) + sum(curr_x(2:end).^2 - 10*cos(4*pi*curr_x(2:end)));
        f2 = g * (1 - (f1/g).^2);
        f(i, :) = [f1, f2];
    end
    f = (f - min(f)) ./ (max(f) - min(f) + 1e-8);
end

function f = calc_Hybrid(x, n_obj)
    n_samples = size(x, 1);
    f = zeros(n_samples, n_obj);
    for i = 1:n_samples
        curr_x = x(i, :);
        zdt_f = calc_ZDT2(curr_x(1:floor(length(curr_x)/2)), n_obj);
        dtlz_f = calc_simplified_DTLZ1(curr_x(floor(length(curr_x)/2)+1:end), n_obj);
        f(i, :) = (zdt_f + dtlz_f) / 2;
    end
    f = (f - min(f)) ./ (max(f) - min(f) + 1e-8);
end

function f = calc_simplified_DTLZ1(x, n_obj)
    n_var = length(x);
    g = 1 + 100 * sum((x - 0.5).^2 - cos(20*pi*(x - 0.5)));
    f = [0.5 * x(1) * (1+g), 0.5 * (1-x(1)) * (1+g)];
end

%% ==============================================
% 辅助函数（优化帕累托判断精度）
% ==============================================
function pareto_idx = is_pareto_optimal(f)
    n_samples = size(f, 1);
    pareto_idx = true(n_samples, 1);
    eps_val = 1e-8; % 加入精度阈值，避免数值误差
    
    for i = 1:n_samples
        if ~pareto_idx(i), continue; end
        for j = 1:n_samples
            if i == j, continue; end
            % 优化支配判断：允许微小数值误差
            if all(f(j, :) <= f(i, :) + eps_val) && any(f(j, :) < f(i, :) - eps_val)
                pareto_idx(i) = false;
                break;
            end
        end
    end
end

function igd = calculate_valid_IGD(pareto_f, true_pareto)
    n_true = size(true_pareto, 1);
    n_pareto = size(pareto_f, 1);
    if n_pareto == 0, igd = 1; return; end
    
    dist_sum = 0;
    for i = 1:n_true
        curr_true = true_pareto(i, :);
        min_dist = min(pdist2(curr_true, pareto_f));
        dist_sum = dist_sum + min_dist;
    end
    igd = dist_sum / n_true;
end

function true_pareto = generate_true_pareto(config)
    n_points = 1000;
    true_pareto = zeros(n_points, 2);
    x = linspace(0, 1, n_points);
    
    if strcmp(config.func, @calc_ZDT2) || strcmp(config.func, @calc_ZDT4)
        true_pareto(:, 1) = x;
        true_pareto(:, 2) = 1 - x.^2;
    elseif strcmp(config.func, @calc_ZDT3)
        true_pareto(:, 1) = x;
        true_pareto(:, 2) = 1 - sqrt(x) - x.*sin(10*pi*x);
    else
        true_pareto(:, 1) = x;
        true_pareto(:, 2) = (1 - x.^2 + 1 - x) / 2;
    end
end

function visualize_results(igd_results, dataset_configs, stat_results, params)
    % 图1：IGD均值对比
    figure(1);
    dataset_names = {stat_results(:).name};
    igd_means = [stat_results(:).mean];
    bar(igd_means, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', dataset_names, 'XTickLabelRotation', 45);
    xlabel('数据集类别');
    ylabel('IGD均值（越小性能越好）');
    title('MOEAs在5类数据集上的IGD均值对比');
    grid on;
    for i = 1:length(igd_means)
        text(i, igd_means(i)+0.001, num2str(igd_means(i), '%.4f'), 'HorizontalAlignment', 'center');
    end
    
    % 图2：帕累托前沿（带多样化解）
    figure(2);
    config = dataset_configs{1};
    [lb, ub, weights] = generate_valid_sample(config, 1);
    [pareto_x, pareto_f] = MOEAs_with_diversity(params, config, lb, ub, weights);
    true_pareto = generate_true_pareto(config);
    plot(true_pareto(:, 1), true_pareto(:, 2), 'r-', 'LineWidth', 2, 'DisplayName', '理论前沿');
    xlabel('目标函数1（0~1）');
    ylabel('目标函数2（0~1）');
    title('小规模结构化数据集帕累托前沿（正常结果）');
    legend('Location', 'best');
    grid on;
end