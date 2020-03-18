function [y_pre, y_post, loss_list, readout_alignment] = ...
    run_session(this, n_trials, x, y_, eta,learning, online_learning)

%{
Run the RNN for a session consisting of many trials.

Parameters:
-----------
n_trials : Number of trials to run the RNN
x : The time-dependent input to the RNN (same for each trial).
y_ : The target RNN output (same for each trial).
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
y : The RNN output.
loss_list : A list with the value of the loss function for each trial.
readout_alignment : The normalized dot product between the vectorized
error feedback matrix and the readout matrix, as in Lillicrap et al
(2016).
%}

if nargin < 7, online_learning = false; end
if nargin < 6, learning = ''; online_learning = false; end
if nargin < 5, eta = [0.1 0.1 0.1]; learning = false; online_learning = false; end

t_max = numel(x);  % number of timesteps
loss_list = [];
readout_alignment = [];

% Flatten the random feedback matrix to check for feedback alignment:
bT_flat = this.b(:);
bT_flat = bT_flat/norm(bT_flat);

for ii = 1:n_trials
    [y, h, u] = this.run_trial(x, y_, eta, learning, online_learning);
    
    % output before and after training
    if ii==1, y_pre = y;
    elseif ii==n_trials, y_post = y; end
    
    % loss
    err = y_ - y;
    loss = 0.5*mean(err.^2);
    loss_list = [loss_list loss];
    
    % alignment
    w_out_flat = (this.w_out)'; w_out_flat = w_out_flat(:);
    w_out_flat = w_out_flat/norm(w_out_flat);
    readout_alignment = [readout_alignment bT_flat'*w_out_flat];
    
    % display
    if mod(ii,1000)==0, fprintf([num2str(ii) '/' num2str(n_trials) '  Loss: ' num2str(loss) '\n']); end
end