function return_bool = switch_lock(e,nom_exp, bool)
%
%  return_bool = switch_lock(e,nom_exp, bool)
%
% When having a list of experiments and multiple CPUs, it is useful to run the whole experiment's batch
% until they are done. We separated experiments between the number crunching part and a figure part,
% allowing to fine tune figures while using strored results.
% 
% Thus we needed a locking system, returning if an experiment if not donce, running or done.
%
% This function implements this by returning:
% -1        if the nom_exp file is not empty and the lock variable is empty : it is done.
% TRUE = 1  if one can extract the lock variable in the nom_exp file and that it set to true: it is running.
% FALSE = 0 in other cases (typically when the nom_exp file is not existing or that the lock variable exists
%               and is FALSE.
% 
% if called with the bool argument, it sets the variable to this value TRUE to lock, FALSE to unlock
%
% in addition, when requesting a lock, the PID and hostname is stored, 
% when checking the lock and getting TRUE, then we return that information to check if a run has not died 


%## Author : Laurent Perrinet <Laurent.Perrinet@incm.cnrs-mrs.fr>
%## This software is distributed under the terms of the GPL


if nargin < 3, % checking the nom_exp file  
    % if no file or bool argument returns FALSE = it's not blocked
    return_bool = 0;
    if exist(nom_exp,'file'),
        load(nom_exp)
        if exist('lock'),
            return_bool = lock;
            if exist('pid'),
                if  lock,
                disp(['File ', nom_exp, ' pid:', num2str(pid),  ' on ', host])
                end
            end
        else,
            return_bool = -1;
        end
    end
else% set the lock variable
    lock = bool;
    if lock,
%        if strcmp(version('-release'),'2007b') | strcmp(version('-release'),'2009a') % the 2 versions of matlab we use in the lab
%            pid = 'dunno how to get the PID in matlab...';
%        else
            pid = getpid();
%        end
       [status host] = unix('hostname');
        save('-v7', nom_exp,'lock','pid','host')
    else
       save('-v7', nom_exp, 'lock')
    end
end
