function [y, h, u] = run_trial(this, x, y_, eta, learning, online_learning)
%{
        Run the RNN for a single trial.

        Parameters:
        -----------
        x : The input to the network. x[t,i] is input from unit i at timestep t.
        y_ : The target RNN output, where y_[t,i] is output i at timestep t.
        eta : A list of 3 learning rates, for w_in, w_rec, and w_out,
            respectively.
        learning : Specify the learning algorithm with one of the following
            strings: 'rtrl', 'bptt', or 'rflo'. If None, run the network without
            learning.
        online_learning : If True (and learning is on), update weights at each
            timestep. If False (and learning is on), update weights only at the
            end of each trial. Online learning cannot be used with BPTT.

        Returns:
        --------
        y : The time-dependent network output. y[t,i] is output i at timestep t.
        h : The time-dependent RNN state vector. h[t,i] is unit i at timestep t.
        u : The inputs to RNN units (feedforward plus recurrent) at each
            timestep.
%}

% neural nonlinearity f and f'
f = @(x) tanh(x);                       % the tanh prevents oveflow
df = @(x) 1./(cosh(10*tanh(x/10)).^2);      % the tanh prevents oveflow

% Boolean shorthands to specify learning algorithm:
rtrl = strcmp(learning, 'rtrl');
bptt = strcmp(learning, 'bptt');
rflo = strcmp(learning, 'rflo');

[eta3, eta2, eta1] = deal(eta(1),eta(2),eta(3));    % learning rates for w_in, w_rec, and w_out
t_max = size(x,1);                                  % number of timesteps

n_in = this.n_in; n_rec = this.n_rec; n_out = this.n_out;
tau_m = this.tau_m;

[dw_in, dw_rec, dw_out] = deal(0, 0, 0);            % changes to weights

u = zeros(t_max, n_rec);                       % input (feedforward plus recurrent)
h = zeros(t_max, n_rec);                       % time-dependent RNN activity vector
h(1,:) = this.h0;                                   % initial state
y = zeros(t_max, n_out);                       % RNN output
err = zeros(t_max, n_out);                     % readout error

%% If rflo, eligibility traces p and q should have rank 2; if rtrl, rank 3:
if rtrl
    p = zeros(n_rec, n_rec, n_rec);
    q = zeros(n_rec, n_rec, n_in);
elseif rflo
    p = zeros(n_rec, n_rec);
    q = zeros(n_rec, n_in);
end

%% initialize
for jj = 1:n_rec
    if rtrl
        q(jj, jj, :) = df(u(1, jj))*x(1,:)/tau_m;
    elseif rflo
        q(jj, :) = df(u(1, jj))*x(1,:)/tau_m;
    end
end

%%
for tt = 1:(t_max-1)
    u(tt+1,:) = this.w_rec*h(tt,:)' + this.w_in*x(tt+1,:)';
    h(tt+1,:) = h(tt,:) + (-h(tt,:) + f(u(tt+1,:)))/tau_m;
    y(tt+1,:) = this.w_out*h(tt+1,:)';
    err(tt+1,:) = y_(tt+1,:) - y(tt+1,:);  % readout error
            
    if rflo
        p = df(u(tt+1,:))'*h(tt,:)/tau_m + (1-1/tau_m)*p;
        q = df(u(tt+1,:))'*x(tt,:)/tau_m + (1-1/tau_m)*q;
    elseif rtrl
%         p = (repmat(df(u(tt+1,:)),[n_rec 1]).*this.w_rec)*p/tau_m + (1-1/tau_m)*p;
%         q = (repmat(df(u(tt+1,:)),[n_rec 1]).*this.w_rec)*p/tau_m + (1-1/tau_m)*p;
        p = squeeze(sum(repmat(repmat(df(u(tt+1,:)),[n_rec 1]).*this.w_rec, [1 1 size(p,2) size(p,3)]).*...
            permute(repmat(p,[1 1 1 n_rec]),[4 1 2 3]),2)) + (1-1/tau_m)*p;
        q = squeeze(sum(repmat(repmat(df(u(tt+1,:)),[n_rec 1]).*this.w_rec, [1 1 size(q,2) size(q,3)]).*...
            permute(repmat(q,[1 1 1 n_rec]),[4 1 2 3]),2)) + (1-1/tau_m)*q;
        for jj = 1:n_rec
            p(jj, jj, :) = squeeze(p(jj, jj, :)) + squeeze((df(u(tt+1, jj))*h(tt,:))'/tau_m);
            q(jj, jj, :) = squeeze(q(jj, jj, :)) + squeeze((df(u(tt+1, jj))*x(tt+1,:))'/tau_m);
        end
    end

    if rflo && online_learning
        dw_out = eta1/t_max*(err(tt+1,:)'*h(tt+1,:));
        dw_rec = eta2*((this.b*err(tt+1,:)')*ones(1,n_rec)).*p/t_max;
        dw_in = eta3*((this.b*err(tt+1,:)')*ones(1,n_in)).*q/t_max;
    elseif rflo && ~online_learning
        dw_out = dw_out + eta1/t_max*(err(tt+1,:)'*h(tt+1,:));
        dw_rec = dw_rec + eta2*((this.b*err(tt+1,:)')*ones(1,n_rec)).*p/t_max;
        dw_in = dw_in + eta3*((this.b*err(tt+1,:)')*ones(1,n_in)).*q/t_max;
    elseif rtrl && online_learning
        dw_out = eta1/t_max*(err(tt+1,:)'*h(tt+1,:));
        dw_rec = eta2*sum(repmat((this.b*err(tt+1,:)'),[1 n_rec n_rec]).*p,1)/t_max;
        dw_in = eta3*sum(repmat((this.b*err(tt+1,:)'),[1 n_rec]).*q,1)/t_max;
    elseif rtrl && ~online_learning
        dw_out = dw_out + squeeze(eta1/t_max*(err(tt+1,:)'*h(tt+1,:)));
        dw_rec = dw_rec + squeeze(eta2*sum(repmat((this.b*err(tt+1,:)'),[1 n_rec n_rec]).*p,1)/t_max);
        dw_in = dw_in + squeeze(eta3*sum(repmat((this.b*err(tt+1,:)'),[1 n_rec]).*q,1)/t_max)';
    end
    
    if online_learning && ~bptt
        this.w_out = this.w_out + dw_out;
        this.w_rec = this.w_rec + dw_rec;
        this.w_in = this.w_in + dw_in;
    end
end

if bptt  % backward pass for BPTT
    z = zeros(t_max, this.n_rec);
    z(end,:) = (this.w_out)'*err(end,:)';
    for tt = t_max:-1:2
        z(tt-1,:) = z(tt,:)*(1 - 1/tau_m);
        z(tt-1,:) = z(tt-1,:) + ((this.w_out)'*err(tt,:)')';
        z(tt-1,:) = z(tt-1,:) + ((z(tt,:).*df(u(tt,:)))*this.w_rec)/tau_m;
        
        % Updates for the weights:
        dw_out = dw_out + eta1*(err(tt,:)'*h(tt,:))/t_max;
        dw_rec = dw_rec + eta2/(t_max*tau_m)*((z(tt,:).*df(u(tt,:)))'*h(tt-1,:));
        dw_in = dw_in + eta2/(t_max*tau_m)*((z(tt,:).*df(u(tt,:)))'*x(tt,:));
    end
end
    
if ~online_learning  % wait until end of trial to update weights
    this.w_out = this.w_out + dw_out;
    this.w_rec = this.w_rec + dw_rec;
    this.w_in = this.w_in + dw_in;
end

end

