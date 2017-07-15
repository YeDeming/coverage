require 'nn'
require 'nngraph'

local attention = {}
function attention.attention(input_size, rnn_size, output_size, dropout)
  local inputs = {}
  local outputs = {}
  table.insert(inputs, nn.Identity()()) -- top_h
  table.insert(inputs, nn.Identity()()) -- fake_region
  table.insert(inputs, nn.Identity()()) -- conv_feat
  table.insert(inputs, nn.Identity()()) -- conv_feat_embed
  table.insert(inputs, nn.Identity()()) -- prev_c


  local h_out = inputs[1]
  local fake_region = inputs[2]
  local conv_feat = inputs[3]
  local conv_feat_embed = inputs[4]
  local prev_c = inputs[5]

  local fake_region = nn.ReLU()(nn.Linear(rnn_size, input_size)(fake_region))
  -- view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
  if dropout > 0 then fake_region = nn.Dropout(dropout)(fake_region) end
  
  local fake_region_embed = nn.Linear(input_size, input_size)(fake_region)

  local h_out_linear = nn.Tanh()(nn.Linear(rnn_size, input_size)(h_out))
  if dropout > 0 then h_out_linear = nn.Dropout(dropout)(h_out_linear) end

  local h_out_embed = nn.Linear(input_size, input_size)(h_out_linear)

  local att_c_linear = nn.Tanh()(nn.Linear(50, input_size)(prev_c))
  if dropout > 0 then att_c_linear = nn.Dropout(dropout)(att_c_linear) end

  local att_c_embed = nn.Linear(input_size, input_size)(att_c_linear)

  local new_h_out_embed = nn.CAddTable()({h_out_embed, att_c_embed})

  local txt_replicate = nn.Replicate(50,2)(new_h_out_embed)      --batch_size * 50 * dim
  -- fake region : s_t
  local img_all = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region), conv_feat})
  local img_all_embed = nn.JoinTable(2)({nn.View(-1,1,input_size)(fake_region_embed), conv_feat_embed}) -- 

  local hA = nn.Tanh()(nn.CAddTable()({img_all_embed, txt_replicate}))
  if dropout > 0 then hA = nn.Dropout(dropout)(hA) end
  local hAflat = nn.Linear(input_size,1)(nn.View(input_size):setNumInputDims(2)(hA))  
  local PI = nn.SoftMax()(nn.View(50):setNumInputDims(2)(hAflat))

  local probs3dim = nn.View(1,-1):setNumInputDims(1)(PI)
  local visAtt = nn.MM(false, false)({probs3dim, img_all})
  local visAttdim = nn.View(input_size):setNumInputDims(2)(visAtt)
  local atten_out = nn.CAddTable()({visAttdim, h_out_linear})

  local h = nn.Tanh()(nn.Linear(input_size, input_size)(atten_out))
  if dropout > 0 then h = nn.Dropout(dropout)(h) end
  local proj = nn.Linear(input_size, output_size)(h)

  local logsoft = nn.LogSoftMax()(proj)
  --local logsoft = nn.SoftMax()(proj)



  -- prev_c h_out
  local h_out_linear_att = nn.Tanh()(nn.Linear(rnn_size, 50)(h_out))
  if dropout > 0 then h_out_linear_att = nn.Dropout(dropout)(h_out_linear_att) end
  local h_out_embed_att = nn.Linear(50, 50)(h_out_linear_att)

  local att_c_linear_att = nn.Tanh()(nn.Linear(50, 50)(prev_c))
  if dropout > 0 then att_c_linear_att = nn.Dropout(dropout)(att_c_linear_att) end

  local att_c_embed_att = nn.Linear(50, 50)(att_c_linear_att)

  local new_h_out_embed_att = nn.CAddTable()({h_out_embed_att, att_c_embed_att})

  local update_gate = nn.Sigmoid()(new_h_out_embed_att);

  local leftc = nn.CMulTable()({prev_c, nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate))});
  local rightc = nn.CMulTable()({PI, update_gate});
  local c = nn.CAddTable()({leftc, rightc});

  table.insert(outputs, logsoft)
  table.insert(outputs, c)


  return nn.gModule(inputs, outputs)
end
return attention