# Chatbot Telmex
import re
import time
from datetime import datetime


# Expresiones Regulares

Promo_RE = r"(?i)(promociones?|promos?|descuentos?|ofertas?|planes? nuevos?|tarifas?|paquetes?|especiales?|baratos?|económicos?|mejores? precios?|qué tienen|qué hay|qué me ofrecen)"

Contratacion_RE = r"(?i)(contratar|alta|nuevo servicio|instalar|quiero contratar|quiero servicio|quiero línea nueva|quiero internet|dar de alta|solicitar|pedir servicio|necesito internet|necesito teléfono|quiero telmex|cómo contrato|requisitos|me interesa)"

Saldo_RE = r"(?i)(saldo|cuánto debo|cu[aá]nto debo|estado de cuenta|cuenta pendiente|mi deuda|debo algo|tengo que pagar|adeudo|balance|cuánto me cobran|qué debo|mi cuenta|revisar cuenta|consultar saldo)"

Pago_RE = r"(?i)(pago|quiero pagar|realizar pago|liquidar|abonar|pagar en línea|pagar en l[ií]nea|forma de pago|formas de pago|dónde pago|donde pago|cómo pago|métodos de pago|pagar con tarjeta|pagar efectivo|bancos|oxxo|seven|tiendas)"

Convenio_RE = r"(?i)(convenio|acuerdo de pago|plan de pago|diferir|aplazar|pago parcial|mensualidades|facilidades|no puedo pagar|pagar poco a poco|dividir pago|pago en partes|ayuda para pagar)"

Recibo_RE = r"(?i)(recibo|factura|dudas de mi recibo|mi factura|cobro|cargos|me llegó caro|recibo alto|recibo elevado|por qué tanto|explicar recibo|desglose|conceptos|qué me cobran|facturación|billing)"

Reporte_RE = r"(?i)(reporte|reportar|fallas?|problema|problemas|no funciona|sin servicio|internet caído|internet ca[ií]do|teléfono muerto|tel[eé]fono muerto|lentitud|lento|no hay señal|sin línea|descompuesto|avería|falla técnica)"

Seguimiento_RE = r"(?i)(seguimiento|estatus de mi reporte|ver avance|estado del reporte|qué pasó con mi folio|mi folio|número de reporte|cuándo vienen|cuándo llega el técnico|avance|progreso|actualización)"

# Módulo de Soporte Técnico
Soporte_RE = r"(?i)(soporte|apoyo|problema|soporte t[eé]cnico|necesito ayuda|ayuda con (mi )?(modem|m[oó]dem|ruteador|router)|configurar (mi )?(wifi|wi-fi|inal[aá]mbrico)|problema con (internet|conexi[oó]n|wifi)|no puedo (entrar|conectar|configurar)|asistencia t[eé]cnica|necesito soporte)"
Modem_RE = r"(?i)(m[oó]dem|modem|router|ruteador|equipo)"
Wifi_RE = r"(?i)(wifi|wi[- ]?fi|inal[aá]mbrico|conexi[oó]n inal[aá]mbrica|señal)"
Config_RE = r"(?i)(configurar|instalar|activar|ajustar|contrase[ñn]a|password|clave)"
Lentitud_RE = r"(?i)(lento|lentitud|tarda|tardado|despacio|baja velocidad|velocidad)"
Desconexion_RE = r"(?i)(se desconecta|pierde conexi[oó]n|cortes|inestable|no se mantiene|se va el internet)"

# Interacciones generales
afirmacion_RE = r"(?i)(sí|si|claro|gracias|por supuesto|ok|dale|okey|está bien|perfecto|excelente|correcto|así es|exacto|de acuerdo)"
salir_RE = r"(?i)(salir|adiós|adios|me equivoqué|perdón|perdon|cancelar|terminar|no|chao|bye|hasta luego|nos vemos|ya no|nada más|eso es todo)"
Saludo_RE = r"(?i)(hola|buenos días|buenas tardes|buenas noches|buen día|buena tarde|buena noche|hey|qué tal|cómo estas|como estas|saludos|que onda|buenas|holi|holaa)"
Despedida_RE = r"(?i)(adiós|adios|hasta luego|nos vemos|chao|bye|hasta pronto|que tengas buen día|gracias y adiós|me voy|ya me voy|hasta la vista|cuídate)"
Ayuda_RE = r"(?i)(ayuda|help|no entiendo|qué puedo hacer|opciones|menú|menu|que opciones hay|qué hay|qué me puedes ayudar|en qué me ayudas|servicios|lista)"
Repetir_RE = r"(?i)(repetir|otra vez|de nuevo|repite|vuelve a decir|no escuché|no entendí|puedes repetir)"

Horarios_RE = r"(?i)(horarios?|qué hora|a qué hora|cuándo abren|cuándo cierran|horario de atención|cuando atienden)"
Sucursales_RE = r"(?i)(sucursales?|oficinas?|dónde están|donde están|direcciones?|ubicación|ubicaciones|centros de atención|tiendas telmex)"
Quejas_RE = r"(?i)(queja|quejas|reclamación|reclamaciones|inconformidad|mal servicio|servicio malo|estoy molesto|no me gusta)"

# Módulo de Pago
Referencia_RE = r"\b\d{8,12}\b"
Monto_RE = r"\$?\s*(\d+(?:\.\d{1,2})?)"
Email_RE = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
Tarjeta_RE = r"(?i)(tarjeta|crédito|credito|débito|debito|visa|mastercard|master card|amex|american express|pagar con tarjeta|plástico)"
Transferencia_RE = r"(?i)(transferencia|transferencia bancaria|banco|depósito|deposito|spei|clabe|interbancaria|app del banco|bancaria|pago digital)"
Efectivo_RE = r"(?i)(efectivo|oxxo|7eleven|7 eleven|farmacias|tienda|super|supermercado|conveniencia|en persona|pago en tienda|en tienda|a la mano|cash|en cualquier tienda)"
Numero_RE = r"\b[1-3]\b"
Domiciliacion_RE = r"(?i)(domiciliar|automático|autom[áa]tico|autom[áa]ticamente|recurrente|permanentemente|cada mes|mensual|pago futuro|futuros pagos|guardar|guardar tarjeta|recordar)"
negacion_RE = r"(?i)(no|negativo|para nada|nunca)"

state_pago = 0
monto_pago = 0
referencia_pago = ""
tarjeta_domiciliada = False

state = 0
Salida = 1

# Ciclo principal
while Salida:
    if state == 0:
        print(" 🤖  ¡Hola! Soy tu asistente virtual Telmex")
        print(" Estoy aquí para ayudarte con lo que necesites.")
        print("\n Menú principal:")
        print(" 1) ⭐ Promociones y ofertas especiales")
        print(" 2) 📝 Contratación de nuevos servicios")
        print(" 3) 💰 Consulta de saldo y estado de cuenta")
        print(" 4) 💳 Formas y lugares de pago")
        print(" 5) 📅 Convenios y facilidades de pago")
        print(" 6) 🧾 Dudas sobre tu recibo o factura")
        print(" 7) 🚨 Reportar fallas o problemas técnicos")
        print(" 8) 🔍 Seguimiento de reportes")
        print(" 9) ⏰ Horarios de atención")
        print("10) 📍 Ubicación de sucursales")
        print("11) 📢 Quejas y sugerencias")
        print("12) 🛠️ Soporte Técnico (modem, WiFi, configuración)")
        opcion = input("\n👉 Escribe tu consulta o número de opción: ")

        if re.findall(Saludo_RE, opcion): 
            state = 0
        elif re.findall(Ayuda_RE, opcion) or re.findall(Repetir_RE, opcion): 
            state = 0
        elif re.findall(Despedida_RE, opcion) or re.findall(salir_RE, opcion): 
            state = 99
        elif opcion.strip() == "1" or re.findall(Promo_RE, opcion): 
            state = 1
        elif opcion.strip() == "2" or re.findall(Contratacion_RE, opcion): 
            state = 2
        elif opcion.strip() == "3" or re.findall(Saldo_RE, opcion): 
            state = 3
        elif opcion.strip() == "4" or re.findall(Pago_RE, opcion): 
            state, state_pago = 4, 40
        elif opcion.strip() == "5" or re.findall(Convenio_RE, opcion): 
            state = 5
        elif opcion.strip() == "6" or re.findall(Recibo_RE, opcion): 
            state = 6
        elif opcion.strip() == "7" or re.findall(Reporte_RE, opcion): 
            state = 7
        elif opcion.strip() == "8" or re.findall(Seguimiento_RE, opcion): 
            state = 8
        elif opcion.strip() == "9" or re.findall(Horarios_RE, opcion): 
            state = 9
        elif opcion.strip() == "10" or re.findall(Sucursales_RE, opcion): 
            state = 10
        elif opcion.strip() == "11" or re.findall(Quejas_RE, opcion): 
            state = 11
        elif opcion.strip() == "12" or re.findall(Soporte_RE, opcion): 
            state = 12
        else: state = 98

    if state == 1:
        print("\nPROMOCIONES ESPECIALES TELMEX")
        print("Te tengo excelentes noticias, estas son nuestras mejores ofertas:")
        print("\n1. Internet Infinitum 200 Mbps - $599/mes por 6 meses")
        print("   (Precio regular $799 - ¡Ahorras $200 mensuales!)")
        print("\n2. Paquete Triple Play Completo - $899/mes por 3 meses")
        print("   Internet + TV + Teléfono (Precio regular $1,199)")
        print("\n3. Internet 100 Mbps + Netflix GRATIS - $499/mes por 12 meses")
        print("   (Precio regular $649 - Netflix incluido sin costo extra)")
        print("\n4. Paquete Empresarial Premium - $1,299/mes por 6 meses")
        print("   Internet 500 Mbps + líneas telefónicas ilimitadas")
        print("Esto es una prueba para comprobar funcionamiento de Git.")
        
        opcion_promo = input("\n¿Alguna de estas promociones te llama la atención?(sí/no): ")
        
        if re.findall(afirmacion_RE, opcion_promo):
            email = input("\n¡Excelente elección! \nPara que un asesor especializado te contacte y te ayude con la contratación, necesito tu correo electrónico: ")
            # Validación básica de email
            if "@" in email and "." in email:
                print(f"\n¡Perfecto! Hemos registrado tu interés.")
                print(f" Un asesor experto se comunicará contigo al correo {email}")
                print(" Tiempo estimado de contacto: máximo 2 horas")
                print(" Te ayudará con todos los detalles y la instalación")
                print("\n ¡Gracias por elegir Telmex! Pronto tendrás el mejor servicio.")
            else:
                print("El formato del correo no parece correcto, pero no te preocupes.")
                print("Hemos registrado tu solicitud y un asesor se comunicará contigo pronto.")
                print("También puedes llamar al 800-123-2222 para más información.")
            state = 90
        elif re.findall(salir_RE, opcion_promo) or re.findall(r"(?i)(no|no gracias|no me interesa|tal vez después)", opcion_promo):
            print("No hay problema, entiendo perfectamente.")
            print("Las promociones estarán disponibles cuando gustes consultarlas.")
            print("Si cambias de opinión, aquí estaré para ayudarte.")
            state = 90
        else:
            print(" No logré entender tu respuesta. Te regreso al menú de promociones.")
            state = 1  # Volver a mostrar promociones

    if state == 2:
        print("\n¡CONTRATACIÓN DE SERVICIOS TELMEX!")
        print("Me da mucho gusto que quieras ser parte de la familia Telmex")
        print("\nPara procesar tu solicitud de contratación necesito algunos datos básicos.")
        print("Un asesor especializado se pondrá en contacto contigo para:")
        print("   • Verificar disponibilidad en tu zona")
        print("   • Explicarte los planes disponibles")
        print("   • Programar la instalación")
        print("   • Resolver todas tus dudas")
        print("\nEl contacto será en las próximas 2 horas hábiles.")
        print("También puedes llamar directamente al 800-123-2222")
        state = 90

    if state == 3:
        print("\nCONSULTA DE SALDO")
        print("Puedo enviarte tu estado de cuenta DETALLADO por correo electrónico.")
        confirmar_envio = input("¿Deseas recibirlo ahora? (sí/no): ")

        if re.findall(afirmacion_RE, confirmar_envio):
            # Solicitar y validar correo electrónico
            email_destino = None
            while True:
                posible = input("Escribe el correo donde quieres recibirlo: ")
                if re.search(Email_RE, posible or ""):
                    email_destino = posible
                    break
                reintentar = input("El formato no parece válido. ¿Intentar de nuevo? (sí/no): ")
                if not re.findall(afirmacion_RE, reintentar or ""):
                    break

            if email_destino:
                periodo = input("Opcional: periodo a consultar (MM/AAAA). Deja vacío para el actual: ")
                print(f"\nPerfecto, enviaré tu estado de cuenta detallado a {email_destino}.")
                print("Contenido del envío:")
                print("   • Saldo actual con fecha de corte")
                print("   • Pagos recibidos y pendientes")
                print("   • Detalle de cargos y bonificaciones")
                print("   • Consumos por servicio (internet/teléfono/TV)")
                print("   • Formas de pago y línea de captura")
                if (periodo or "").strip():
                    print(f"   • Periodo solicitado: {periodo}")
                state = 90
            else:
                print("\nDe acuerdo, no realizaré el envío por correo.")
                print("Si lo deseas más tarde, vuelve a pedirme 'consulta de saldo'.")
                state = 90

        elif re.findall(salir_RE, confirmar_envio) or re.findall(r"(?i)(no|luego|despu[eé]s)", confirmar_envio or ""):
            print("De acuerdo. Si más tarde deseas el envío, pídeme 'consulta de saldo'.")
            state = 90
        else:
            print("No logré entender tu respuesta. Vamos a intentarlo de nuevo.")
            state = 3

    if state == 4:
        if state_pago == 40:
            print("\nPAGO DE SERVICIO")
            referencia_input = input(" Para empezar, por favor ingresa tu número de referencia de Telmex: ")
            referencia_match = re.search(Referencia_RE, referencia_input)
            if referencia_match:
                referencia_pago = referencia_match.group()
                state_pago = 41
            else:
                print("El número de referencia no es válido. Debe contener entre 8 y 12 dígitos.")
            
        elif state_pago == 41:
            monto_input = input("Ingresa el monto a pagar: ")
            
            monto_match = re.search(Monto_RE, monto_input)
            if monto_match:
                try:
                    monto_pago = float(monto_match.group(1))
                    state_pago = 42
                except ValueError:
                    print("Por favor, ingresa un monto válido (ejemplo: 499.50)")
            else:
                print("Por favor, ingresa un monto válido (ejemplo: 499.50)")
                
        elif state_pago == 42:
            print(f"\nConfirmado: Vas a pagar ${monto_pago:.2f} para la referencia {referencia_pago}")
            print("¿Cómo deseas realizar tu pago?")
            print("1. Tarjeta de crédito/débito")
            print("2. Transferencia bancaria")
            print("3. Efectivo en tiendas")
            
            metodo_input = input("\nSelecciona una opción (1-3) o describe tu método preferido: ")
            
            if re.findall(Tarjeta_RE, metodo_input) or re.search(Numero_RE, metodo_input) and re.search(r"\b1\b", metodo_input):
                print("\nSerás redirigido a nuestro portal seguro de pagos con tarjeta...")
                time.sleep(2)
                print("Pago procesado exitosamente")
                print(f"Número de transacción: PAG-{datetime.now().strftime('%Y%m%d%H%M%S')}")
                
                # Preguntar por domiciliación para pagos futuros
                print("\n¿Te gustaría domiciliar este método de pago para futuros pagos?")
                print("Así tus pagos se realizarán automáticamente cada mes.")
                domiciliar_input = input("(sí/no): ")
                
                if re.findall(afirmacion_RE, domiciliar_input) or re.findall(Domiciliacion_RE, domiciliar_input):
                    print("\n¡Excelente! Hemos registrado tu tarjeta para pagos automáticos.")
                    print("Cada mes, tu pago se realizará automáticamente con esta tarjeta.")
                    print("Recibirás una notificación por correo antes de cada cargo.")
                    tarjeta_domiciliada = True
                elif re.findall(negacion_RE, domiciliar_input):
                    print("\nDe acuerdo, no domiciliaremos tu tarjeta.")
                    print("Podrás realizar tus pagos manualmente cada mes.")
                else:
                    print("\nNo entendí tu respuesta. No domiciliaremos tu tarjeta por ahora.")
                    print("Puedes configurar la domiciliación más adelante si lo deseas.")
                
                state_pago = 43
            elif re.findall(Transferencia_RE, metodo_input) or re.search(Numero_RE, metodo_input) and re.search(r"\b2\b", metodo_input):
                print("\nDatos para transferencia:")
                print("Banco: BBVA")
                print("CLABE: 012 180 00123456789 1")
                print("Beneficiario: TELMEX SA DE CV")
                print(f"Referencia: {referencia_pago}")
                print(f"Monto: ${monto_pago:.2f}")
                print("\nUna vez realizada la transferencia, tu pago se reflejará en 24-48 horas.")
                state_pago = 43
            elif re.findall(Efectivo_RE, metodo_input) or re.search(Numero_RE, metodo_input) and re.search(r"\b3\b", metodo_input):
                print("\nPuedes pagar en efectivo en:")
                print("- OXXO")
                print("- 7-Eleven")
                print("- Farmacias del Ahorro")
                print("- Supermercados participantes")
                print(f"\nReferencia: {referencia_pago}")
                print(f"Monto: ${monto_pago:.2f}")
                print("Tu pago se procesará en cuanto se realice el depósito.")
                state_pago = 43
            else:
                print("No pude identificar tu método de pago. Por favor selecciona 1, 2 o 3.")
                
        elif state_pago == 43:
            email_input = input("\n¿Deseas recibir el comprobante de pago por correo electrónico? (sí/no): ")
            
            if re.findall(afirmacion_RE, email_input):
                while True:
                    email_pago = input("Ingresa tu correo electrónico: ")
                    email_match = re.search(Email_RE, email_pago)
                    if email_match:
                        print(f"\nComprobante enviado a {email_pago}")
                        print("Revisa tu bandeja de entrada en los próximos minutos.")
                        break
                    else:
                        print("El formato del correo no es válido. Por favor, ingresa un correo válido.")
            
            print("\n¡Gracias por tu pago!")
            print("Tu servicio continuará activo sin interrupciones.")
            
            if tarjeta_domiciliada:
                print("\nRecuerda que hemos registrado tu tarjeta para pagos automáticos.")
                print("No necesitarás realizar el pago manualmente cada mes.")
            
            print("Si tienes más preguntas, estoy aquí para ayudarte.")
            state_pago = 0
            state = 90
            
    if state == 5:
        print("\nCONVENIOS Y FACILIDADES DE PAGO")
        print("Entendemos que a veces necesitas flexibilidad para pagar")
        print("\nTe ofrecemos estas opciones:")
        print("   • Convenio a 3 meses SIN INTERESES")
        print("   • Convenio a 6 meses SIN INTERESES")
        print("   • Pago parcial con compromiso de liquidación")
        print("\nBeneficios:")
        print("   • Mantienes tu servicio activo")
        print("   • Sin afectación a tu historial crediticio")
        print("   • Proceso rápido y sencillo")
        print("\nPara solicitar tu convenio llama al 800-123-2222")
        state = 90

    if state == 6:
        print("\nEXPLICACIÓN DE TU RECIBO")
        print("Te ayudo a entender cada concepto de tu factura:")
        print("\nCONCEPTOS PRINCIPALES:")
        print("   • Renta básica: Costo fijo mensual del servicio")
        print("   • Consumo adicional: Llamadas extras o datos excedentes")
        print("   • IVA: Impuesto al Valor Agregado (16%)")
        print("   • Servicios adicionales: Netflix, HBO, etc.")
        print("\n¿Tienes dudas sobre algún cargo específico?")
        print("Envía una foto de tu recibo al WhatsApp 55-1234-5678")
        print("O llama al 800-123-2222 para explicación detallada")
        state = 90

    if state == 7:
        print("\nREPORTE DE FALLAS")
        print("Lamento mucho los inconvenientes que estás experimentando.")
        confirmar = input("\n¿Deseas que levante un reporte técnico ahora? (sí/no): ")

        if re.findall(afirmacion_RE, confirmar):
            # Referencia 8-12 dígitos obligatoria
            ref = ""
            ref_digits = ""
            while True:
                ref = input("\nPerfecto. Indica tu número Telmex o número de cuenta (8 a 12 dígitos): ")
                # Cancelación explícita (evita falsos positivos por 'no' dentro de palabras)
                if re.search(r"(?i)^\s*(salir|cancelar|terminar|ya no|nada m[aá]s|eso es todo|no)\s*$", ref or ""):
                    print("De acuerdo, no levantaré el reporte por ahora.")
                    state = 90
                    break
                ref_digits = re.sub(r"\D", "", ref or "")
                if 8 <= len(ref_digits) <= 12:
                    print("Referencia válida ✓")
                    break
                else:
                    print("La referencia debe tener entre 8 y 12 dígitos. Intenta nuevamente o escribe 'cancelar'.")
            if state == 90:
                # Usuario canceló durante la referencia
                continue

            direccion = input("¿En qué domicilio se presenta la falla? (calle y colonia/ciudad): ")
            descripcion = input("Describe brevemente la falla (sin servicio, lento, ruido en línea, etc.): ")
            horario = input("¿Cuál es el mejor horario para visita? (mañana/tarde/noche o 9-14/14-18): ")

            # Validación de medio de contacto
            contacto_valor = ""
            while True:
                contacto_tipo = input("¿Prefieres contacto por teléfono o por correo?: ")
                # Cancelación explícita (evita que 'teléfono' coincida por 'no')
                if re.search(r"(?i)^\s*(salir|cancelar|terminar|ya no|nada m[aá]s|eso es todo|no)\s*$", contacto_tipo or ""):
                    print("De acuerdo, no levantaré el reporte por ahora.")
                    state = 90
                    break
                if re.search(r"(?i)tel[eé]fono|cel|m[oó]vil|whats|wa", contacto_tipo or ""):
                    telefono = input("Indícame tu número de contacto (10 dígitos): ")
                    tel_digits = re.sub(r"\D", "", telefono or "")
                    if len(tel_digits) == 10:
                        contacto_valor = tel_digits
                        print("Número válido ✓")
                        break
                    else:
                        print("El teléfono debe tener exactamente 10 dígitos.")
                        continue
                else:
                    correo = input("Indícame tu correo de contacto (formato nombre@dominio.com): ")
                    if re.search(Email_RE, correo or ""):
                        contacto_valor = correo
                        print("Correo válido ✓")
                        break
                    else:
                        print("El correo no es válido. Intenta nuevamente o escribe 'cancelar'.")
                        continue
            if state == 90:
                # Usuario canceló durante contacto
                continue

            print("\nResumen de tu reporte:")
            print(f"   • Referencia (línea/cuenta): {ref_digits}")
            print(f"   • Domicilio: {direccion}")
            print(f"   • Falla: {descripcion}")
            print(f"   • Horario preferente: {horario}")
            print(f"   • Contacto: {contacto_valor}")

            confirmar_final = input("\n¿Confirmas levantar el reporte con esta información? (sí/no): ")
            if re.findall(afirmacion_RE, confirmar_final):
                folio = "TLX-" + str(datetime.now().year) + "-" + str(int(time.time()))[-6:]
                print("\n¡Listo! He registrado tu reporte de falla:")
                print(f"   • Fecha y hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
                print(f"   • Folio de seguimiento: {folio}")
                print("   • Prioridad: Alta")
                print("\nPROCESO DE ATENCIÓN:")
                print("   • Diagnóstico remoto: Inmediato")
                print("   • Revisión técnica: Máximo 24 horas")
                print("   • Visita técnica (si es necesaria): 24-48 horas")
                print("\nRecibirás SMS o llamada con actualizaciones del progreso.")
                print("Para seguimiento inmediato puedes llamar al 800-123-2222 y proporcionar tu folio.")
                state = 90
            elif re.findall(salir_RE, confirmar_final) or re.findall(r"(?i)(no|no gracias|mejor después|luego)", confirmar_final):
                print("Entendido, no levantaré el reporte por ahora. Si lo deseas, puedo ayudarte en otro tema.")
                state = 90
            else:
                print("No logré entender tu respuesta. Te regreso al menú de reportes.")
                state = 7

        elif re.findall(salir_RE, confirmar) or re.findall(r"(?i)(no|no gracias|no por ahora|luego)", confirmar):
            print("No hay problema. Si cambias de opinión, puedo levantar el reporte cuando gustes.")
            state = 90
        else:
            print("No logré entender tu respuesta. Te regreso al menú de reportes.")
            state = 7

    if state == 8:
        print("\nSEGUIMIENTO DE TU REPORTE")
        folio_in = input("Para ayudarte, indícame tu folio TLX-AAAA-###### o tu número de contacto/cuenta: ")

        folio = None
        folio_match = re.match(r"(?i)^\s*TLX-(\d{4})-(\d{6})\s*$", folio_in or "")
        if folio_match:
            folio = f"TLX-{folio_match.group(1)}-{folio_match.group(2)}"
        else:
            digits = re.sub(r"\D", "", folio_in or "")
            if len(digits) >= 7:
                folio = f"TLX-{datetime.now().year}-{digits[-6:].zfill(6)}"
            else:
                print("No logré identificar un folio o referencia válido.")
                print("Intenta nuevamente con tu folio o un número de cuenta/teléfono asociado.")
                state = 8
                
        if folio:
            print("\nConsultando el estatus de tu reporte, por favor espera...")
            time.sleep(1)

            seed = int(re.sub(r"\D", "", folio)[-6:])
            slot = "9:00 AM - 1:00 PM" if seed % 2 == 0 else "2:00 PM - 6:00 PM"
            techs = [
                "Juan Pérez (ID: 12345)",
                "María López (ID: 23456)",
                "Carlos Ruiz (ID: 34567)",
                "Ana Gómez (ID: 45678)"
            ]
            tecnico = techs[seed % len(techs)]

            diag = "Completado ✓" if seed % 3 != 0 else "En curso"
            solucion = "Identificada ✓" if seed % 4 != 0 else "En análisis"
            visita_dia = "Hoy" if datetime.now().hour < 12 else "Mañana"

            print("\nINFORMACIÓN DEL REPORTE:")
            print(f"   • Folio: {folio}")
            print("   • Estatus: EN PROCESO")
            print(f"   • Diagnóstico: {diag}")
            print(f"   • Solución: {solucion}")
            print(f"   • Técnico asignado: {tecnico}")
            print("\n VISITA PROGRAMADA:")
            print(f"   • Fecha: {visita_dia}")
            print(f"   • Horario estimado: {slot}")
            print("   • El técnico te llamará 30 min antes")

            print("\nOpciones de seguimiento:")
            print("   1) Reprogramar visita")
            print("   2) Actualizar medio de contacto")
            print("   3) Agregar notas para el técnico")
            print("   4) Cancelar el reporte")
            print("   5) Volver al inicio")

            accion = input("¿Qué deseas hacer? (1/2/3/4/5 o escribe la opción): ")

            if re.search(r"(?i)^(1|reprogram)", accion or ""):
                nueva_fecha = input("Nueva fecha (DD/MM/AAAA): ")
                nuevo_horario = input("Horario preferente (mañana/tarde o 9-13/14-18): ")
                print("\nHe solicitado la reprogramación de tu visita:")
                print(f"   • Folio: {folio}")
                print(f"   • Fecha: {nueva_fecha}")
                print(f"   • Horario: {nuevo_horario}")
                print("Recibirás confirmación por SMS/Email en breve.")
                state = 90

            elif re.search(r"(?i)^(2|actualizar|contacto)", accion or ""):
                prefer = input("¿Prefieres actualizar teléfono o correo?: ")
                if re.search(r"(?i)tel|cel|m[oó]vil|whats|wa", prefer or ""):
                    nuevo_tel = input("Nuevo teléfono (10 dígitos): ")
                    print("Contacto actualizado. Usaremos este teléfono para avisos de tu reporte.")
                else:
                    nuevo_mail = input("Nuevo correo: ")
                    print("Contacto actualizado. Usaremos este correo para avisos de tu reporte.")
                state = 90

            elif re.search(r"(?i)^(3|nota|agregar)", accion or ""):
                nota = input("Escribe la nota que quieres que el técnico vea: ")
                print("\nNota agregada a tu folio correctamente.")
                print(f"   • Folio: {folio}")
                state = 90

            elif re.search(r"(?i)^(4|cancel)", accion or ""):
                conf = input("¿Seguro que deseas cancelar el reporte? (sí/no): ")
                if re.findall(afirmacion_RE, conf or ""):
                    print("\nTu reporte ha sido cancelado. Si el problema persiste, puedes generar uno nuevo en cualquier momento.")
                    print(f"   • Folio cancelado: {folio}")
                    state = 90
                elif re.findall(salir_RE, conf or ""):
                    print("No cancelaré el reporte. Seguimos monitoreando su avance.")
                    state = 90
                else:
                    print("No logré entender tu respuesta. Mantendré el reporte activo.")
                    state = 90

            elif re.search(r"(?i)^(5|volver|inicio|menu|menú)", accion or ""):
                state = 90
            else:
                print("No logré entender tu respuesta. Te regreso al menú de seguimiento.")
                state = 8

    if state == 9:
        print("\n HORARIOS DE ATENCIÓN TELMEX")
        print(" ATENCIÓN TELEFÓNICA:")
        print("   • Lunes a Domingo: 24 horas")
        print("   • Número: 800-123-2222")
        print("\n SUCURSALES:")
        print("   • Lunes a Viernes: 9:00 AM - 6:00 PM")
        print("   • Sábados: 9:00 AM - 2:00 PM")
        print("   • Domingos: Cerrado")
        print("\nATENCIÓN EN LÍNEA:")
        print("   • Chat: 24/7 en telmex.com")
        print("   • WhatsApp: 55-1234-5678 (24 horas)")
        state = 90

    if state == 10:
        print("\n SUCURSALES TELMEX")
        print(" PRINCIPALES UBICACIONES:")
        print("   • Centro: Av. Juárez 123, Col. Centro")
        print("   • Norte: Av. Insurgentes Norte 456")
        print("   • Sur: Av. División del Norte 789")
        print("   • Oriente: Av. Zaragoza 321")
        print("   • Poniente: Av. Observatorio 654")
        print("\nPara encontrar la sucursal más cercana:")
        print("   • Visita: telmex.com/sucursales")
        print("   • Llama al: 800-123-2222")
        print("   • WhatsApp: 55-1234-5678")
        state = 90

    if state == 11:
        print("\nQUEJAS Y SUGERENCIAS")
        print("Lamento mucho que hayas tenido una mala experiencia")
        print("Tu opinión es muy importante para nosotros.")
        print("\nCANALES PARA TU QUEJA:")
        print("   • Teléfono: 800-123-2222 (Opción 9)")
        print("   • Email: quejas@telmex.com")
        print("   • WhatsApp: 55-1234-5678")
        print("   • Presencial: Cualquier sucursal")
        print("\n TIEMPO DE RESPUESTA:")
        print("   • Acuse de recibo: Inmediato")
        print("   • Investigación: 3-5 días hábiles")
        print("   • Resolución: Máximo 15 días hábiles")
        print("\n Nos comprometemos a resolver tu situación satisfactoriamente.")
        state = 90

    if state == 12:
        print("\n🛠️ SOPORTE TÉCNICO TELMEX")
        print("Puedo ayudarte con algunos consejos para problemas comunes.")
        problema = input("\nPor favor, dime cuál es tu problema (ej. modem, wifi, configuración, lentitud, desconexión): ")

        if re.findall(Modem_RE, problema):
            print("\n🔧 Consejo: Reinicia tu módem desconectándolo 10 segundos y vuelve a conectarlo.")
        elif re.findall(Wifi_RE, problema):
            print("\n📡 Consejo: Verifica que estés conectado a la red correcta y revisa la contraseña.")
        elif re.findall(Config_RE, problema):
            print("\n⚙️ Consejo: Ingresa en tu navegador a http://192.168.1.254 (usuario: admin, contraseña: la etiqueta del módem).")
        elif re.findall(Lentitud_RE, problema):
            print("\n🐢 Consejo: Revisa que no haya muchos dispositivos conectados al mismo tiempo.")
        elif re.findall(Desconexion_RE, problema):
            print("\n🔌 Consejo: Verifica que los cables del módem estén bien conectados y cambia el canal Wi-Fi.")
        else:
            print("\n🤔 No identifiqué el problema exacto. Te doy ayuda básica, pero quizá necesites soporte real.")

        feedback = input("\n¿Este consejo resolvió tu problema? (sí/no): ")
        if re.findall(afirmacion_RE, feedback):
            print("\n¡Excelente! Me alegra haberte ayudado. 😊")
            state = 90
        else:
            print("\nEntiendo. Para atención más especializada comunícate con:")
            print("   • Teléfono: 800-123-2222")
            print("   • WhatsApp: 55-1234-5678")
            print("   • Chat en línea: telmex.com")
            state = 90

    # Estado de Continuación    
    if state == 90:
        opcion = input("\n ¿Hay algo más en lo que te pueda ayudar? (sí/no): ")
        
        # Verificar despedidas también en este punto
        if re.findall(Despedida_RE, opcion) or re.findall(salir_RE, opcion):
            state = 99
        elif re.findall(afirmacion_RE, opcion):
            print("¡Perfecto! Con mucho gusto te sigo ayudando")
            state = 0
        # También verificar si saluda de nuevo
        elif re.findall(Saludo_RE, opcion):
            print("¡Hola de nuevo! Te ayudo con mucho gusto")
            state = 0
        else:
            print(" No logré entender tu respuesta, pero no te preocupes.")
            print("Te regreso al menú principal para que puedas elegir otra opción.")
            state = 0

    # Estado de error
    if state == 98:
        print("\n Disculpa, no logré entender exactamente qué necesitas.")
        print("Pero no te preocupes, estoy aquí para ayudarte ")
        print("\n💡 Puedes preguntarme sobre cualquiera de estos temas:")
        print(" • Promociones y ofertas especiales")
        print(" • Contratación de nuevos servicios")
        print(" • Consulta de saldo y estado de cuenta")
        print(" • Formas y lugares de pago")
        print(" • Convenios y facilidades de pago")
        print(" • Dudas sobre recibo o factura")
        print(" • Reportar fallas o problemas")
        print(" • Seguimiento de reportes")
        print(" • Horarios de atención")
        print(" • Ubicación de sucursales")
        print(" • Quejas y sugerencias")
        print("\n Intenta escribir tu consulta de otra manera, por ejemplo:")
        print("   'quiero ver promociones' o 'tengo una falla' o 'dónde pago'")
        state = 0

    if state == 99:
        print("\n¡Gracias por contactar a Telmex!")
        print("Fue un placer atenderte hoy. 📞")
        print("Recuerda que estamos disponibles 24/7 para ayudarte.")
        print("¡Hasta luego y que tengas un excelente día! 🌟")
        Salida = 0
