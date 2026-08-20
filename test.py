import ast
output= ["Consumer / Shopping Issue", "I am sorry to hear that you received defective food and are facing difficulty getting a refund.\n\nBased on the Consumer Protection Act rules for handling consumer complaints and defective goods, here are the steps you can take:\n\n1. **Contact the Grievance Officer:** You can report your issue directly to the platforms designated Grievance Officer, who is required to acknowledge your grievance within 48 hours.\n2. **File a Consumer Complaint:** If your issue is not resolved, you have the right to file a formal complaint regarding defective goods before the District, State, or National Consumer Disputes Redressal Commission, depending on the value of your claim.\n\nIf the situation remains unresolved or you require further assistance in filing a complaint, you may consider contacting a legal aid clinic or a legal professional." ]

# ml = ast.literal_eval(output)

types = """"
"Police / Arrest Issue" → maps to BNSS sections on FIR filing, arrest rights, bail
"Consumer / Shopping Issue" → maps to Consumer Protection Act (defective products, refunds, e-commerce fraud)
"Workplace Harassment" → maps to POSH Act (sexual harassment at workplace)
"Domestic Violence / Safety at Home" → maps to Protection of Women from Domestic Violence Act
"Government Info Request" → maps to RTI Act (how to file, timelines, appeals)
"Child Safety Concern" → maps to POCSO Act
"General / Not Sure" (fallback) → skips pre-filled context, goes to open freeform cha
"""

print(output[1])