import os
from jinja2 import Environment, FileSystemLoader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

base_dir = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(base_dir))

template = env.get_template("police_comp_template.txt")

rti_data = {
    "applicant_name": "Ramesh Kumar",
    "address": "123 MG Road, Chennai, Tamil Nadu",
    "department": "Municipal Corporation",
    "info_requested": "Status of road repair complaint filed on 01/08/2026, including expected completion date and officer assigned",
    "date": "20/08/2026"
}

consumer_data = {
    "complainant_name": "Priya Sharma",
    "company_name": "XYZ Electronics Pvt Ltd",
    "product_service": "Samsung Refrigerator (Model SR-450)",
    "purchase_date": "15/06/2026",
    "issue_description": "Refrigerator stopped cooling within 2 months of purchase. Company refused free repair despite valid warranty and has not responded to 3 follow-up calls.",
    "desired_resolution": "Full replacement of the product or complete refund of Rs. 32,000",
    "date": "20/08/2026"
}

police_data = {
    "complainant_name": "Arjun Nair",
    "police_station": "T. Nagar Police Station, Chennai",
    "incident_type": "Theft of two-wheeler",
    "incident_datetime": "18/08/2026, approximately 9:30 PM",
    "incident_location": "Parking area near Panagal Park, T. Nagar",
    "incident_description": "My motorcycle (Registration No. TN09XXXX) was stolen from the parking area while I was at a nearby shop for approximately 20 minutes.",
    "witnesses": "None identified at the time; CCTV footage may be available from nearby shops",
    "date": "20/08/2026"
}

# ---- Step 4: Fill the template with data ----
filled_text = template.render(**police_data)

print(filled_text)  # sanity check — see the filled text in terminal

# ---- Step 5: Convert filled text to PDF ----
def text_to_pdf(text, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 60
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 60
    c.save()

output_path = os.path.join(base_dir, f"output_files/police_{police_data['complainant_name']}.pdf")
text_to_pdf(filled_text, output_path)

print(f"PDF saved at: {output_path}")