classdef LPrelu_2_Layer < nnet.layer.Layer
    %     properties (Learnable)
    %         Cut
    %     end
    methods
        function layer = LPrelu_2_Layer(numChannels, name)
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "LPrelu_2_Layer with " + numChannels + " channels";
            
            % layer.Cut = 10;
        end
        
        function Z = predict(layer, X)
            
            Z =   (0 < X & X <= 5).*X + (5 < X & X <= 8).*(5 + 0.05.*X) + (8 < X).*(5.4 + 0.02.*X);
            % Z =   (0 < X & X < 4).*X + (4 < X).*4; %clipped_ReLU
            %Z =   (0 < X & X <= layer.Cut).*X + (layer.Cut < X).*(layer.Cut + 0.05.*X);
        end
        
    end
end