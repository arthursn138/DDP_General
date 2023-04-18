function [x, u, J, lambda, dlambda, alpha, k_u, K_u, ii, iter_succ, total_cost] =...
    disc_ddp_alg(ddp_2nd_order, f_dyn, run_cost, term_cost, ddp_par, ubar, xbar, opt_par);

    % First order discrete ddp (iLQR) without constraints
    % Regularization part is by Yuichiro Aoyama
    dt = ddp_par.dt; N = ddp_par.N; n = ddp_par.n; m = ddp_par.m; 
    x0 = ddp_par.x0; xf = ddp_par.xf;
    iter = opt_par.iter; toler = opt_par.toler; lambda = opt_par.lambda;
    dlambda = opt_par.dlambda; lambdaFactor = opt_par.lambdaFactor; 
    lambdaMax = opt_par.lambdaMax; lambdaMin = opt_par.lambdaMin;
    
    % Compute nominal running and terminal costs
    L=0; %initialze
    for jj = 1:N-1
        L = L + run_cost(xbar(:,jj), ubar(:,jj), false);
    end

    [L_f] = term_cost(xbar(:,end),false);
    total_cost = L + L_f;
    
    % Initialize the costs for eah iteration and expected decrease in cost
    J = zeros(1, iter);
    dJ_exp = [0 0];
    iter_succ = [];

    % Discrete DDP (iLQR) algorithm
    for ii = 1:iter
        J(ii) = total_cost;                

        % Backward pass: from N to 1; for N:
        
        % Get V,Vx,Vxx
        [L_f, L_f_x, L_f_xx] = term_cost(xbar(:,end), true);
        V_plus = L_f; Vx_plus = L_f_x; Vxx_plus = L_f_xx;
        
        % Initialize ff gain and fb gain
        k_u = zeros(m,N); % ff gain
        K_u = zeros(m,n,N); % fb gain

        for jj = N-1:-1:1
            f_xk = f_dyn.fx(xbar(:,jj), ubar(:,jj));
            f_uk = f_dyn.fu(xbar(:,jj), ubar(:,jj));
            [~, L_xk, L_uk, L_xuk, L_xxk, L_uuk] = run_cost(xbar(:,jj), ubar(:,jj), true);
            Qx = L_xk + f_xk'*Vx_plus;
            Qu = L_uk + f_uk'*Vx_plus;
            Qxx = L_xxk + f_xk'*Vxx_plus*f_xk;
            Qxu = L_xuk + f_xk'*Vxx_plus*f_uk; Qux = Qxu';
            Quu = L_uuk + f_uk'*Vxx_plus*f_uk;
            Quu_reg = Quu + lambda*eye(m);

            % Regularization check
            if any(isnan(Quu_reg))
                error('Quu_reg has NaN in ddp\n')
            end
            if lambda>=lambdaMax    
                    fprintf('Regularizer is too large');
                    break
            end     
            if min(eig(Quu_reg))<0
                fprintf(1, 'Quu ndf at %d . make regularizer large \n',jj);
                dlambda=max(dlambda*lambdaFactor,lambdaFactor);
                lambda=max(lambda*dlambda,lambdaMin);               
                break            
            end

            k_u(:,jj) = -Quu_reg\Qu; % ff gain
            K_u(:,:,jj) = -Quu_reg\Qux;  % fb gain
            
            Vx_plus = Qx + K_u(:,:,jj)'*Quu*k_u(:,jj) + K_u(:,:,jj)'*Qu + Qxu*k_u(:,jj); %Qx + Qxu*l(:,k);
            Vxx_plus = Qxx + K_u(:,:,jj)'*Quu*K_u(:,:,jj) + K_u(:,:,jj)'*Qux + Qxu*K_u(:,:,jj);
            
            % Vxx is positive definite
            Vxx_plus = 0.5 * (Vxx_plus + Vxx_plus'); 
            
            % For expected decrease in cost:
            dJ_exp = dJ_exp + [k_u(:,jj)'*Qu  0.5*k_u(:,jj)'*Quu*k_u(:,jj)];
        end
        
        % Forward pass:
        fwdPassDone =0;
        
        % Initilizations 
        x = zeros(n,N);
        x(:,1) = x0;
        u = zeros(m,N-1);
        deltax = x(:,1) - xbar(:,1);
        
        % for regularization of the feedforward control (line search):
        alpha0 = 10.^linspace(0,-3,11);
        for alpha = alpha0 
            % Forward propagation
            L = 0;
            for jj = 1:N-1
                deltau = alpha*k_u(:,jj) + K_u(:,:,jj)*deltax;
                u(:,jj) = ubar(:,jj) + deltau;
                x(:,jj + 1) = f_dyn.f(x(:,jj),u(:,jj));
                deltax = x(:,jj + 1) - xbar(:,jj + 1);
                L = L + run_cost(x(:,jj),u(:,jj),false);
            end
            L_f = term_cost(x(:,end),true);
            total_cost_new = L + L_f;

            % Actual decrease in cost
            dcost = total_cost-total_cost_new;

            % Expected decrease in cost
            expected = - alpha*(dJ_exp(1) + alpha*dJ_exp(2));

            % Ratio of the actual decrease to the expected decrease
            z = dcost/expected;
            if expected < 0
               z = sign(dcost);
               warning('negative expected reduction in cost!!');
            end               
            if z > 0
               fwdPassDone = 1;
               break;
            end
        end
        
        % Regularization and cost reduction (from Yuichiro Aoyama) 
        if fwdPassDone 
            if norm(x(2,end)-xf(2)) <= 0.1
                iter_succ = [iter_succ ii];
            end
            % Decrease lambda
            dlambda   = min(dlambda / lambdaFactor, 1/lambdaFactor);
            lambda    = lambda * dlambda * (lambda > lambdaMin);        
            conv_flag = 0;
            if abs(dcost) < toler           
                conv_flag = 1;
            end
            
            % Update
            total_cost = total_cost_new;          
            xbar = x; ubar = u;
            if conv_flag, fprintf('convergence of DDP dcost: %.3f\n',dcost);
                break; 
            end
        
        else
            dlambda   = max(dlambda * lambdaFactor, lambdaFactor);
            lambda    = max(lambda * dlambda, lambdaMin);
            if isinf(total_cost)
                fprintf('diverged, try with larger lambda: %2f\n',lambda)
            else
                fprintf('Not enough cost reduction make lambda larger,lambda:%2f\n',lambda);
            end
            
            if lambda > lambdaMax
                 norm_costgrad = 1;
                 x = xbar;
                 u = ubar;
                 fprintf('\nEXIT: lambda > lambdaMax\n');
                 break;
            end
        end       
    end
    iter_succ = min(iter_succ);
    J = J(1:ii);
end