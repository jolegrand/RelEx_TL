local inputwords = torch.Tensor()
local inputentities = torch.Tensor()

function test(network, data, params)
   local fusr = {}
   for i=1,#params.fusr do
      fusr[params.fusr[i]]=i
   end
   
   local pathdata
   if params.mobius then
      pathdata = "/home/runuser/corpus/semevalData/"
   else
      pathdata = "/home/joel/Bureau/loria/corpus/Semeval2013DDI/"
   end

   local tstfile, filename, goldfilename
   if data.corpus=="medline" or data.corpus=="drugbank" or data.corpus=="full" then
      local d = data.corpus=="medline" and 'MedLine' or (data.corpus=='drugbank' and 'DrugBank' or 'Full')
      goldfilename = pathdata .. "/Test/DDI_Extraction/" .. d .. "/"
      filename = "task9.2_UC3M_1.txt"
      tstfile = io.open(params.rundir .. "/" .. filename, 'w')
   end
      
   local timer = torch.Timer()
   timer:reset()

   network:evaluate()
   
   local criterion = nn.ClassNLLCriterion()

   if params.dropout~=0 then
      if params.dp==1 or params.dp==3 or params.dp==4 then
	 network.dropout:evaluate()
      end
      if params.dt==2 or params.dt==3 then
	 for i=1, #network.RNNs do
	    network.RNNs[i]:evaluate()
	 end
      end
   end
   
   local cost = 0
   local nforward = 0

   local confusion_matrix = torch.Tensor(#data.relationhash, #data.relationhash):fill(0)
   
   local precision_recall = {}
   for i=1,#data.relationhash do
      precision_recall[i] = {totalpos=0, truepos=0, falsepos=0}
   end
   
   for idx=1,data.size do
      --print(idx .. " " .. data.words[idx]:size(1) .. "/" .. data.size)
      -- print(data.words[idx]:size(1))
      -- print(data.trees[idx])
      --print(data.ids[idx])
      --printw(data.words[idx], data.wordhash)
      if idx%1000==0 then print(idx .. " / " .. data.size) end
      collectgarbage()
      
      local words = data.words[idx]
      if (params.arch==2 or params.dp==2 or params.dp==3 or params.batch~=0  or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then words = words:view(1,words:size(1)) end
      
      if params.arch==3 and data.entities.nent(data, idx)>=2 then network:getGraph(data.trees[idx], data) end

      if data.entities.nent(data, idx)<2 then else
	 local n = data.entities.nent(data, idx)
	 nforward = nforward + ((n * (n-1))/2)
      end      

      if false and params.batch~=0 then
	 local n = data.entities.nent(data, idx)
	 local nf = ((n * (n-1))/2)
	 inputwords = words:view(1,words:size(1)):expand(nf, words:size(1))
	 inputentities:resize(nf, words:size(1)):fill(-1)
	 local c = 0
	 for ent1=1,data.entities.nent(data, idx) do
	    for ent2=ent1+1,data.entities.nent(data, idx) do
	       c = c + 1
	       inputentities[c]:copy( data.entities.getent(data, idx, ent1, ent2) )
	    end
	 end
	 local input = {inputwords}
	 table.insert(input, inputentities)
	 local output = network:forward(input)
	 local max, indice = output:max(1)
	 tstfile:write(data.ids[idx] .. "|" .. data.ids[idx] .. ".e" .. ent1-1 .. "|" .. data.ids[idx] .. ".e" .. ent2-1 .. "|" .. (max[1]~=1 and "1|" or "0|") .. data.relationhash[indice[1]] .. "\n")
      else
	 for ent1=1,data.entities.nent(data, idx) do
	    for ent2=ent1+1,data.entities.nent(data, idx) do
	       if is_related(params, data, idx, ent1, ent2) then
		  --print("relation between " .. ent1 .. " and " .. ent2 .. " (" .. data.relations:isrelated(idx, ent1, ent2) .. ")")
		  local entities = data.entities.getent(data, idx, ent1, ent2)
		  if (params.arch==2 or params.dp==2 or params.dp==3 or params.batch~=0  or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then entities = entities:view(1, entities:size(1)) end
		  
		  local input = {words}
		  if params.tfsz~=0 then table.insert(input, data.entities.getenttags(data, idx)) end
		  if params.pfsz~=0 then table.insert(input, data.pos[idx]) end
		  if params.rdfsz~=0 then
		     table.insert(input, data.get_relative_distance(entities, 1))
		     table.insert(input, data.get_relative_distance(entities, 2))
		  end
		  if params.dtfsz~=0 and params.dt==1 then
		     local dtf = data.deptypes[idx]
		     if (params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then dtf = dtf:view(1, dtf:size(1)) end
		     table.insert(input, dtf)
		  end
		  table.insert(input, entities)
		  if params.dtfsz~=0 and params.dt==2 then
		     local dtf = data.deptypes[idx]
		     if (params.dp==2 or params.dp==3 or params.batch~=0 or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5)) then dtf = dtf:view(1, dtf:size(1)) end
		     input = {input, dtf}
		  end

		  if params.arch==5 then
		     --print(entities)
		     data.trees2[idx]:LCA(3,4,entities,1,2)
		     data.trees2[idx]:set_tab_sontags(1)
		     --datas[datacorpus].trees2[idx]:print_tree("")
		     --io.read()
		  end
		  
		  local output
		  if params.arch==1 or params.arch==3 or params.arch==6 then
		     output = network:forward(input, data.corpus)
		  else
		     output = network:forward(data.trees2[idx], input, data.corpus)
		  end
		  if params.arch==4 or params.arch==5 then network.treelstm:clean(data.trees2[idx]) end
		  
		  
		  local target = data.relations:isrelated(idx, ent1, ent2)

		  cost = cost + criterion:forward(output, target)
		  
		  if (params.dp==2 or params.dp==3 or params.batch~=0  or params.rnn=="lstm" or params.rnn=="cnn") and (not (params.arch==4 or params.arch==5))  then output = output[1] end

		  local max, indice = output:max(1)
		  indice = indice[1]

		  --print(target .. " " .. indice)
		  
		  if data.corpus=="medline" or data.corpus=="drugbank" or data.corpus=="full" then
		     tstfile:write(data.ids[idx] .. "|" .. data.ids[idx] .. ".e" .. ent1-1 .. "|" .. data.ids[idx] .. ".e" .. ent2-1 .. "|" .. (indice~=1 and "1|" or "0|") .. data.relationhash[indice] .. "\n")
		  end
		  
		  local class = data.relations:isrelated(idx, ent1, ent2)
		  --if data.relationhash[class]=="int" then io.read() end
		  precision_recall[class].totalpos = precision_recall[class].totalpos +1
		  if class==indice then
		     precision_recall[indice].truepos = precision_recall[indice].truepos+1
		  else
		     -- print("error")
		     if false then
			printw(words, data.wordhash)
			print(data.relationhash[class] .. " but classified as " .. data.relationhash[indice]) 
			io.read()
		     end
		     precision_recall[indice].falsepos = precision_recall[indice].falsepos + 1
		  end
		  confusion_matrix[class][indice] = confusion_matrix[class][indice] + 1
		  
	       end
	    end
	 end
      end
   end

   local t = timer:time().real
   print(string.format('test corpus processed in %.2f seconds (%.2f sentences/s)', t, data.size/t))
   
   cost = cost/nforward

   local class_to_consider
   if data.corpus=="medline" or data.corpus=="drugbank" or data.corpus=="full" then
      if fusr["full"] then class_to_consider = {5} else class_to_consider = {2,3,4,5} end
   elseif data.corpus=="snpphena" then
      class_to_consider = {2,6}
   elseif data.corpus=="ppi" or data.corpus=="LLL" or data.corpus=="AIMed" or data.corpus=="HPRD50" or data.corpus=="IEPA" or data.corpus=="BioInfer" then
      class_to_consider = {2}
   elseif data.corpus=="ADE" then
      class_to_consider = {2}
   elseif data.corpus=="EUADR_drug_disease" or data.corpus=="EUADR_drug_target" or data.corpus=="EUADR_target_disease" then
      class_to_consider = {2}
   elseif data.corpus=="reACE" then
      class_to_consider = {2,3,4,5,6,7}
   elseif data.corpus=="PGxCorpus" then
      if params.onerelation then
	 class_to_consider = {2}
      else      
	 class_to_consider = {2,3,4,5,6,7,8,9,10}
      end
   else
      error("unknown corpus")
   end
   
   --computing evaluation measures
   --macro-average (avg min and presicion over all categories)
   local recalls, precisions = {}, {}
   local macro_R, macro_P = 0, 0
   for k,i in pairs(class_to_consider) do
      --print(data.relationhash[i])
      local a = precision_recall[i].truepos
      local b = precision_recall[i].totalpos
      recalls[i] = (a==0 and b==0 and 0 or a/b)
      --print("a " .. a .. " b " .. b .. " R " .. recalls[i])
      local a = precision_recall[i].truepos
      local b = precision_recall[i].truepos + precision_recall[i].falsepos
      precisions[i] = (a==0 and b==0 and 0 or a/b)
      --print("a " .. a .. " b " .. b .. " P " .. precisions[i] .. " fp " .. precision_recall[i].falsepos)
            
      macro_R = macro_R + ((recalls[i]==recalls[i]) and recalls[i] or 0)
      macro_P = macro_P + ((precisions[i]==precisions[i]) and precisions[i] or 0)
   end
   macro_R = macro_R / (#class_to_consider)
   macro_P = macro_P / (#class_to_consider)
   local macro_f1score = (2 * macro_R * macro_P) / (macro_R + macro_P)
   macro_f1score = macro_f1score==macro_f1score and macro_f1score or 0
   
   --micro average precision (sum truepos, falsepos, totalpos and compute P and R)
   local _truepos, _falsepos, _totalpos = 0, 0, 0
   for k,i in pairs(class_to_consider) do
      _truepos = _truepos + precision_recall[i].truepos
      _totalpos = _totalpos + precision_recall[i].totalpos
      _falsepos = _falsepos + precision_recall[i].falsepos
   end
   local a = _truepos
   local b = _totalpos
   --print("a " .. a .. " b " .. b)
   local micro_R = (a==0 and b==0 and 0 or a/b) 
   local a = _truepos
   local b = _truepos + _falsepos
   --print("a " .. a .. " b " .. b)
   local micro_P = (a==0 and b==0 and 0 or a/b)
   local micro_f1score = (2 * micro_R * micro_P) / (micro_R + micro_P)
   micro_f1score = micro_f1score==micro_f1score and micro_f1score or 0

   
   --print(data.relationhash)
   print("\t\tP\tR")
   for k,i in pairs(class_to_consider) do
      print("Class " .. i .. ":\t" .. string.format('%.2f',precisions[i]) .. "\t" .. string.format('%.2f',recalls[i])) 
   end

   print(confusion_matrix)
   
   if false and (data.corpus=="medline" or data.corpus=="drugbank" or data.corpus=="full") then
      print("Detection and classification of DDI")
      print("tp\tfp\tfn\ttotal\tprec\trecall\tF1\n")
      print("nc\tnc\tnc\tnc\t" .. string.format("%.4f", micro_P) .. "\t" .. string.format("%.4f", micro_R) .. "\t" .. string.format("%.4f", micro_f1score))
      print("\nSCORES FOR DDI TYPE")
      for i=2,#data.relationhash do
	 local tp = confusion_matrix[i][i]
	 local fp = confusion_matrix:narrow(2,i,1):sum()-tp
	 local fn = precision_recall[i].totalpos-tp
	 local f1 = (2*recalls[i]*precisions[i])/(recalls[i]+precisions[i])
	 f1 = f1==f1 and f1 or 0
	 print("Scores for ddi with type " .. data.relationhash[i])
	 print("tp\tfp\tfn\ttotal\tprec\trecall\tF1")
	 print(tp .. "\t" .. fp .. "\t" .. fn .. "\t"  .. "nc" .. "\t" .. string.format("%.4f", precisions[i]) .. "\t" .. string.format("%.4f",recalls[i]) .. "\t" .. string.format("%.4f",f1))
      end
      print("\nMACRO-AVERAGE MEASURES FOR DDI CLASSIFICATION:\n \tP\tR\tF1")
      print(" \t" .. string.format("%.4f", macro_P) .. "\t" .. string.format("%.4f", macro_R) .. "\t" .. string.format("%.4f", macro_f1score))

      
      local d = assert(io.popen("pwd")):read()
      local cmd = "cd " .. params.rundir .. "; java -jar " .. d .. "/evaluateDDI.jar " .. goldfilename .. " " .. filename
      print(cmd)
      local f = assert(io.popen(cmd))
      local txt = f:read('*all')
      f:close()
      print(txt)
      
      local _micro_P = txt:match('Detection and classification of DDI.*(%d[,.]%d+).*%d[,.]%d+.*%d[,.]%d+[ %t]*.*SCORES FOR DDI TYPE')
      local _micro_R = txt:match('Detection and classification of DDI.*%d[,.]%d+.*(%d[,.]%d+).*%d[,.]%d+[ %t]*.*SCORES FOR DDI TYPE')
      local _micro_f1score = txt:match('Detection and classification of DDI.*%d[,.]%d+.*%d[,.]%d+.*(%d[,.]%d+)[ %t]*.*SCORES FOR DDI TYPE')
      _micro_P = _micro_P and _micro_P:gsub(",","%.") or 0
      _micro_R = _micro_R and _micro_R:gsub(",","%.") or 0
      _micro_f1score = _micro_f1score and _micro_f1score:gsub(",","%.") or 0
      
      local _macro_P = txt:match('MACRO%-AVERAGE.*(%d[,.]%d+).*%d[,.]%d+.*%d[,.]%d+[ %t]*')
      local _macro_R = txt:match('MACRO%-AVERAGE.*%d[,.]%d+.*(%d[,.]%d+).*%d[,.]%d+[ %t]*')
      local _macro_f1score = txt:match('MACRO%-AVERAGE.*%d[,.]%d+.*%d[,.]%d+.*(%d[,.]%d+)[ %t]*')
      _macro_P = _macro_P and _macro_P:gsub(",","%.") or 0
      _macro_R = _macro_R and _macro_R:gsub(",","%.") or 0
      _macro_f1score = _macro_f1score and _macro_f1score:gsub(",","%.") or 0

      assert(math.abs(macro_P - _macro_P)<0.01, macro_P .. " " .. _macro_P .. " " .. math.abs(macro_P - _macro_P))
      assert(math.abs(micro_P - _micro_P)<0.01, micro_P .. " " .. _micro_P .. " " .. math.abs(micro_P - _micro_P))
      assert(math.abs(macro_R - _macro_R)<0.01, macro_R .. " " .. _macro_R .. " " .. math.abs(macro_R - _macro_R))
      assert(math.abs(micro_R - _micro_R)<0.01, micro_R .. " " .. _micro_R .. " " .. math.abs(micro_R - _micro_R))
      assert(math.abs(macro_f1score - _macro_f1score)<0.01, macro_f1score .. " " .. _macro_f1score .. " " .. math.abs(macro_f1score - _macro_f1score))
      assert(math.abs(micro_f1score - _micro_f1score)<0.01, micro_f1score .. " " .. _micro_f1score .. " " .. math.abs(micro_f1score - _micro_f1score))
      
   end

   return (macro_P or 0), (macro_R or 0), (macro_f1score or 0), cost, micro_P, micro_R, micro_f1score      
end
