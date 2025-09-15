classdef ExpressionEvaluator < handle
    % ExpressionEvaluator - Manage user typed expression evaluation
    %
    % This class is intended to manage evaluation of expressions that a
    % user may enter into part of an app or graphical interface. This class
    % may have source data containing a structure, table, or object
    % associated with the expression, where the fields/variables/properties
    % may be used in the expression without reference to the original
    % structure/table/object.
    %
    % Syntax:
    %       obj = ExpressionEvaluator()
    %
    % Notes:
    %   1. Recommended enhancement: let SourceData be a cell containing
    %   multiple valid data sources.
    %
    
    % Copyright 2018 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: nhowes $
    %   $Revision: 194 $  $Date: 2019-08-05 16:16:01 -0400 (Mon, 05 Aug 2019) $
    % ---------------------------------------------------------------------
    
    
    %% Properties
    
    properties
        SourceData %The source data for the expression (object, struct, table)
        SearchWorkspace (1,1) logical = true; %Indicate whether to include base and caller workspace variables
        IgnoreCase (1,1) logical = true; %Indicate whether variable names are case sensitive
    end %properties
    
    
    
    %% Public Methods
    methods
        
        function [result,statusOk,message,expression] = eval(obj,expression)
            % Evaluate an expression
            
            % Validate inputs
            narginchk(2,2);
            validateattributes(expression,{'char','string'},{});
            
            % Default output
            statusOk = true;
            message = '';
            result = [];
            
            % Get a list of valid variable names from the source data
            if istable(obj.SourceData)
                
                sourceVarNames = obj.SourceData.Properties.VariableNames;
                
            elseif isstruct(obj.SourceData)
                
                sourceVarNames = fieldnames(obj.SourceData);
                
            elseif isobject(obj.SourceData)
                
                sourceVarNames = properties(obj.SourceData);
                
            end %if
            
            % Trim the expression
            expression = strtrim(expression);
            
            % Remove any quoted char/string from expression. Leave a token
            % to re-insert later.
            pattern = '((?<=\<'')[^'']+(?=''))|((?<=\<")[^"]+(?="))';
            stringsFound = regexp(expression,pattern,'match');
            expNoStrings = regexprep(expression,pattern,'<_strtoken_>');
            
            % Look for identifiers in the expression
            %Old pattern: '\<[A-z]\w*'
            pattern = '\<[A-Za-z]\w*';
            if obj.IgnoreCase
                identifiersFound = regexp(expNoStrings,pattern,'match');
                identifiersFound = lower(identifiersFound);
            else
                identifiersFound = regexpi(expNoStrings,pattern,'match');
            end
            identifiersFound = unique(identifiersFound);
            
            % Keep track of which identifiers have been replaced, and their
            % new values
            identifierSearchComplete = false(size(identifiersFound));
            identifierReplacement = cell(size(identifiersFound));
            
            % Which identifiers from SourceData are used in the expression?
            thisIdentFound = false(size(identifiersFound));
            replacementIdx = zeros(size(identifiersFound));
            for idx = 1:numel(replacementIdx)
                if obj.IgnoreCase
                    thisVarMatch = strcmpi(identifiersFound{idx},sourceVarNames);
                else
                    thisVarMatch = strcmp(identifiersFound{idx},sourceVarNames);
                    %thisIdentFound(idx) = any( strcmp(identifiersFound{idx},sourceVarNames) );
                end
                if any(thisVarMatch)
                    replacementIdx(idx) = find(thisVarMatch,1);
                    thisIdentFound(idx) = true;
                end
            end
            
            %thisVarNames = sourceVarNames(varFoundIdx(isFound));
            replacementVars = sourceVarNames(replacementIdx(thisIdentFound));
            identifierReplacement(thisIdentFound) = strcat('obj.SourceData','.',replacementVars);
            identifierSearchComplete(thisIdentFound) = true;
            
            % If SearchWorkspace is enabled
            if obj.SearchWorkspace && ~all(identifierSearchComplete)
                
                % Get workspace vars
                workSpace.Base = evalin('base','who');
                workSpace.Caller = evalin('caller','who');
                
                % Loop on each that needs raplacing
                for idx = 1:numel(identifierSearchComplete)
                    if ~identifierSearchComplete(idx)
                        
                        thisVar = identifiersFound{idx};
                        
                        if obj.IgnoreCase
                            isBaseMatch = strcmpi(workSpace.Base,thisVar);
                            isCallerMatch = strcmpi(workSpace.Caller,thisVar);
                            % If multiple found, use the case sensitive one
                            if sum(isBaseMatch)>1
                                isBaseMatch = strcmp(workSpace.Base,thisVar);
                            end
                            if sum(isCallerMatch)>1
                                isCallerMatch = strcmp(workSpace.Caller,thisVar);
                            end
                        else
                            isBaseMatch = strcmp(workSpace.Base,thisVar);
                            isCallerMatch = strcmp(workSpace.Caller,thisVar);
                        end
                        
                        % Get the replacements
                        if any(isCallerMatch)
                            wsVarName = workSpace.Caller(isCallerMatch);
                            Assign.(wsVarName) = evalin('caller',wsVarName);
                            identifierSearchComplete(idx) = true;
                            identifierReplacement{idx} = ['Assign.' wsVarName];
                        elseif any(isBaseMatch)
                            wsVarName = workSpace.Base{isBaseMatch};
                            Assign.(wsVarName) = evalin('base',wsVarName);
                            identifierSearchComplete(idx) = true;
                            identifierReplacement{idx} = ['Assign.' wsVarName];
                        end
                        
                    end %if
                end %for
                
            end %if obj.SearchWorkspace...
            
            % Check if any identifiers are functions. If so, ignore it.
            if ~all(identifierSearchComplete)
                isFunction = cellfun(@(x)exist(x)>1,...
                    identifiersFound(~identifierSearchComplete) ); %#ok<EXIST>
                identifierSearchComplete(~identifierSearchComplete) = isFunction;
            end
            
            % Verify all identifiers have been found
            if ~all(identifierSearchComplete)
                result = [];
                statusOk = false;
                message = sprintf(...
                    'Undefined attribute or function: (%s).',...
                    strjoin(identifiersFound(~identifierSearchComplete),','));
                if nargout > 1
                    return
                else
                    error('ExpressionEvaluator:invalidExpression',message) %#ok<SPERR>
                end
            end
            
            % Replace the found identifiers
            toReplace = identifierSearchComplete &...
                ~cellfun(@isempty,identifierReplacement);
            if obj.IgnoreCase
                args = {'ignorecase'};
            else
                args = {};
            end
            expression = regexprep(expNoStrings,...
                strcat('\<',identifiersFound(toReplace),'\>'),...
                identifierReplacement(toReplace), args{:});
            
            % Replace any quoted char/string we removed %TODO @ROBYN FOR
            % REVIEW 
            for i = 1 : numel(stringsFound)
                expression = regexprep(expression,'<_strtoken_>',stringsFound(i),'once');
            end
            % Evaluate the resulting expression
            try
                result = eval(expression);
                %validateattributes(index,{'logical'},{'vector' 'numel' AttributeSize});
            catch err
                statusOk = false;
                message = sprintf('Expression failed: %s', err.message);
            end
            
        end %function
        
    end %methods
    
    
    
    %% Get/Set Methods
    methods
        
        function set.SourceData(obj,value)
            if ~isobject(value)
                validateattributes(value,{'struct','table'},{})
            end
            obj.SourceData = value;
        end
        
    end %methods
    
end % classdef