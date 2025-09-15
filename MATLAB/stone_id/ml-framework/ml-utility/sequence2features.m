classdef sequence2features < matlab.mixin.SetGet
    %SEQUENCE2FEATURES Engineer lagged features for machine learning  from sequence data
    %
    %
    %   N.C.Howes
    %   Copyright 2021 The MathWorks Inc.
    
    properties (SetAccess = private, GetAccess = public )
        Output  table      
    end

    properties ( Dependent )
        TSVariables
        GroupingVariables
        Sequences table
        Offsets
        LookaheadVariables
    end
    
    properties
        Type (1,1) string {mustBeMember(Type,["Lag", "Lead"])} = "Lag"
        Verbose (1,1) logical = false
    end
    
    properties ( Access = protected )
        Input
        GroupingVariables_ (1,:) string = ""
        TSVariables_ (1,:) string = ""
        Sequences_
        Offsets_ (1,:) double = 1:2
        LookaheadVariables_  
    end
    
    
    methods
        function obj = sequence2features( sequences, varargin )
            %SEQUENCE2FEATURES Construct an instance of this class
            
            if nargin > 0
                
                obj.Sequences = sequences;
                
                if ~isempty( varargin )
                    set(obj, varargin{:})
                end
                
                obj.apply()
                
            end %if nargin > 0
            
        end %Sequence2Features
    end
    
    methods
        function set.LookaheadVariables( obj, value )
            obj.LookaheadVariables_ = sort(value);
        end
        
        function value = get.LookaheadVariables( obj )
            value = obj.LookaheadVariables_;
        end
        
        
        function set.Offsets( obj, value )
            obj.Offsets_ = sort(value);
        end
        
        function value = get.Offsets( obj )
            value = obj.Offsets_;
        end
        
        function set.Sequences( obj, value )
            obj.TSVariables_ = value.Properties.VariableNames;
            obj.Sequences_ = value;
        end
        
        function value = get.Sequences( obj )
            if isempty( obj.TSVariables )
                value = obj.Sequences_;
            else
                value = obj.Sequences_(:, obj.TSVariables);
            end
        end
        
        function set.TSVariables( obj, value )
            obj.TSVariables_ = value;
        end
        
        function value = get.TSVariables( obj )
            value = setdiff(obj.TSVariables_, obj.GroupingVariables, "stable");
        end
        
        function set.GroupingVariables( obj, value )
            
            validNames = obj.Sequences_.Properties.VariableNames;
            
            if ~any( ismember(validNames, value) )
                error("Not a valid grouping variable")
            end
            
            obj.GroupingVariables_ = value;
        end
        
        function value = get.GroupingVariables( obj )
            value = obj.GroupingVariables_;
        end
        
    end
    
    methods ( Access = protected )
        
        function apply( obj )
            %APPLY Summary of this method goes here
            
            sequences       = obj.Sequences;
            offsets         = obj.Offsets;
            gVars           = obj.GroupingVariables;
            timeSeriesVars  = obj.TSVariables;
            
            if ~( gVars == "")
                [groups, iD] = findgroups( obj.Sequences_(:,gVars) );
            else
                groups = ones( height(obj.Sequences),1 );
            end
            
            grpList = unique( groups );

            maxLag      = max( offsets );
            nSequences  = numel( offsets ) + 1;    
            nInputs     = size( sequences,2 );
            nSamples    = nnz( groups == grpList(1) ) - maxLag;
            nGroup      = numel(grpList);
               
           
            groupSeries = cell( nGroup , 1 );
            
            
            for iGroup = 1 : nGroup
                
                inputs = sequences( groups == grpList(iGroup),: );
  
                if obj.Verbose == true
                    fprintf( "Engineering Features Sequence %d of %d.\n", iGroup, nGroup )
                end

                offsetValues = [ 0 offsets ];
                
                if maxLag > ( size( inputs,1 )-1 )
                    %   warning( "Offset must not exceed input series length-1." )
                    continue
                end

                lagSeries  = cell(1, nInputs);
                varNames   = cell(1, nInputs);
                for iInput = 1 : nInputs

                    thisSeries =  inputs{:,iInput};
                    
                    if isnumeric( thisSeries )
                        lags = zeros( nSamples , nSequences );
                    elseif isstring( thisSeries ) || ischar( thisSeries) || iscellstr( thisSeries )
                        lags = strings( nSamples , nSequences );
                    elseif isdatetime( thisSeries )
                        lags = NaT( nSamples , nSequences );
                    elseif iscategorical( thisSeries )
                        lags = repmat(thisSeries(1), nSamples, nSequences ) ;
                        lags(:) = '<undefined>';
                    else
                        error( 'Unhandled class' )
                    end
                     
                    if obj.Type == "Lag"
                        lags(:,1) = thisSeries( 1+maxLag : end );
                    else
                        lags(:,1) = thisSeries( 1 : end-maxLag );
                    end
                    
                    cnt = 1;
                    for iLag = offsets
                        cnt = cnt+1;
                        
                        if obj.Type == "Lag"
                            lags(:,cnt) = thisSeries(  1 + (maxLag-iLag) : end - iLag   );
                        else
                            lags(:,cnt) = thisSeries( 1 + iLag : end - ( maxLag-iLag ) ) ;
                        end
                        
                    end %for iLag
                    
                    if isdatetime( lags )
                    lags.Format = thisSeries.Format;
                    end
                        
                    baseName = timeSeriesVars( iInput )+ ": " + "t";
                    
                    if obj.Type == "Lag"
                        qualifier = "-";
                    else
                        qualifier = "+";
                    end
                     
                    varNames{ iInput } = [baseName baseName + qualifier + offsetValues(2:end) ];
                    lagSeries{ iInput } = num2cell( lags ); %[lagSeries num2cell(transpose( lags ))];
                      
                end %for iInput

                  groupSeries{ iGroup } = cat(2,lagSeries{:});

            end %for iGroup
 
            toKeep  = cellfun(@(x)~isempty(x), groupSeries);
            groupSeries = groupSeries( toKeep, : );

            features = cell2table( vertcat( groupSeries{:} ), ...
                     'VariableNames', [varNames{:}]);
            
            if ~( obj.GroupingVariables == "" )
                iD = iD(toKeep,:);
                nRep = cellfun(@(x)size(x,1),groupSeries(:,1));
                grp = repelem(iD,nRep,1);
                obj.Output = [grp features ];
            else
                obj.Output =  features;
            end
            
            obj.Output.Properties.RowNames = "Obs" + (1 : height(obj.Output) );
     
            vars = string( obj.Output.Properties.VariableNames );
           
            list = vars( ~cellfun(@isempty, regexp( vars, ": t$" ) ) );
           
            obj.Output(:, list( ~ contains( list, [obj.LookaheadVariables] ) ) ) = [];
            
        end %apply
    end
    
    methods
        function value = table( obj )
            value = obj.Output;
        end
        
        function head( obj )
           head(obj.Output) 
        end
        
    end %public
    
    
    methods ( Static )
        function value = result( sequences, varargin )
            
            sF = sequence2features( sequences, varargin{:} );
            
            value = sF.table;
            
            delete( sF )
        end %result
    end
end


